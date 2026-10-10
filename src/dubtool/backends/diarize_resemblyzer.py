"""Multi-speaker diarization from ungated parts only.

Exists because pyannote isn't always reachable: pyannote.audio 4.x pulls a
PLDA transform from pyannote/speaker-diarization-community-1, whose gate is
approved by hand rather than instantly like speaker-diarization-3.1's and
segmentation-3.0's, so "accept the terms and go" can turn into an
open-ended wait. Downgrading to pyannote 3.x, which needs only the
instant-gated models, doesn't work either: 3.x calls
`torchaudio.AudioMetaData`, removed in torchaudio 2.11, which is why
pyproject pins 4.0.7.

So this backend uses nothing gated and nothing new: Resemblyzer for voice
embeddings, webrtcvad for speech detection, scikit-learn for clustering --
all already dependencies. It deliberately embeds with the *same*
Resemblyzer encoder stages/voice_bank.py uses to recognize a voice across
runs, and clusters at a distance derived from that module's similarity
threshold, so "the same speaker" means one thing everywhere in the
pipeline rather than two subtly different things.

How it works: webrtcvad marks speech; Resemblyzer embeds overlapping 1.6s
windows across it; agglomerative clustering groups those windows by voice;
adjacent windows sharing a label merge into one turn.

What it gives up against pyannote: no overlapping-speech handling (two
characters talking at once land in one turn, assigned to whichever voice
dominates), and turn boundaries land on window edges (~0.4s) instead of
pyannote's neural segmentation, which is frame-accurate. It is far better
than SingleSpeakerStub, which assigns every line in an episode to one
voice, and worse than pyannote when pyannote is actually available --
which is the whole tradeoff it exists to offer.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from dubtool.types import Segment

log = logging.getLogger(__name__)

_SAMPLE_RATE = 16000          # Resemblyzer and webrtcvad both require this
_VAD_FRAME_MS = 30            # webrtcvad accepts only 10/20/30
_MIN_SPEECH_COVERAGE = 0.5    # a window must be at least half speech to cluster
_MIN_TURN_DURATION = 0.4      # drop turns too short to transcribe usefully
_MAX_BRIDGED_PAUSE = 0.35     # pauses shorter than this are mid-sentence, not turn changes


class ResemblyzerDiarizer:
    def __init__(
        self,
        distance_threshold: float = 0.40,
        vad_aggressiveness: int = 2,
        max_speakers: int = 12,
    ):
        # Cosine distance at which two windows stop counting as one voice.
        # 0.40 from sweeping a real separated-vocals clip: the speaker count
        # falls 26 -> 14 -> 5 -> 2 -> 1 across 0.20..0.50 with no stable
        # plateau anywhere, so there is no "correct" value to find here --
        # 0.40 sits mid-range, and pass --num-speakers whenever the cast size
        # is known, which clusters to a fixed k and avoids this guess
        # entirely. Treat auto-detected counts as a hint, not a measurement.
        # Deliberately *not* derived from voice_bank.py's 0.75 similarity
        # threshold, which is a different measurement: that one compares two
        # clean, hand-picked reference clips, while these are 1.6s windows cut
        # wherever they happen to fall, often mid-word and over music. Reusing
        # 0.75 (a 0.25 distance) split a 37s two-hander into 20 voices.
        self.distance_threshold = distance_threshold
        self.vad_aggressiveness = vad_aggressiveness
        self.max_speakers = max_speakers

    def diarize(self, audio_path: Path, num_speakers: int | None = None) -> list[Segment]:
        wav = self._load_16k_mono(audio_path)
        duration = len(wav) / _SAMPLE_RATE
        if duration <= 0:
            return []

        speech_mask = self._speech_mask(wav)
        if not speech_mask.any():
            log.warning("no speech detected by VAD; treating the clip as one speaker")
            return [Segment(start=0.0, end=duration, speaker_id="SPEAKER_00")]

        windows, embeddings = self._embed_windows(wav, speech_mask)
        if len(windows) == 0:
            log.warning("no window had enough speech to embed; treating the clip as one speaker")
            return [Segment(start=0.0, end=duration, speaker_id="SPEAKER_00")]
        if len(windows) == 1:
            return [Segment(start=windows[0][0], end=windows[0][1], speaker_id="SPEAKER_00")]

        labels = self._cluster(embeddings, num_speakers)
        segments = self._assign_to_speech_regions(speech_mask, windows, labels, duration)
        log.info(
            "diarized %.1fs into %d turn(s) across %d speaker(s)",
            duration, len(segments), len({s.speaker_id for s in segments}),
        )
        return segments

    # --- stages ---------------------------------------------------------

    def _load_16k_mono(self, audio_path: Path) -> np.ndarray:
        # Deliberately not resemblyzer.preprocess_wav: in 0.1.4 that trims
        # long silences with no way to switch it off, which shifts every
        # timestamp after the first trimmed gap. Diarization output is
        # nothing but timestamps, so it has to be the untrimmed signal.
        import librosa

        wav, _ = librosa.load(str(audio_path), sr=_SAMPLE_RATE, mono=True)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 0:
            wav = wav / peak * 0.95   # Resemblyzer expects roughly normalized input
        return wav.astype(np.float32)

    def _speech_mask(self, wav: np.ndarray) -> np.ndarray:
        """Per-sample boolean speech mask from webrtcvad."""
        import webrtcvad

        vad = webrtcvad.Vad(self.vad_aggressiveness)
        frame_len = int(_SAMPLE_RATE * _VAD_FRAME_MS / 1000)
        pcm = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)

        mask = np.zeros(len(wav), dtype=bool)
        for start in range(0, len(pcm) - frame_len + 1, frame_len):
            frame = pcm[start:start + frame_len]
            try:
                if vad.is_speech(frame.tobytes(), _SAMPLE_RATE):
                    mask[start:start + frame_len] = True
            except Exception:  # pragma: no cover - malformed tail frame
                continue
        return mask

    def _embed_windows(
        self, wav: np.ndarray, speech_mask: np.ndarray
    ) -> tuple[list[tuple[float, float]], np.ndarray]:
        """Resemblyzer's own partial windows (1.6s, ~0.77s hop), keeping only
        those with enough speech in them to represent a voice."""
        from resemblyzer import VoiceEncoder

        encoder = VoiceEncoder()
        _, partial_embeds, wav_slices = encoder.embed_utterance(wav, return_partials=True)

        windows: list[tuple[float, float]] = []
        kept: list[np.ndarray] = []
        for embed, sl in zip(partial_embeds, wav_slices):
            chunk = speech_mask[sl]
            if chunk.size and chunk.mean() >= _MIN_SPEECH_COVERAGE:
                windows.append((sl.start / _SAMPLE_RATE, sl.stop / _SAMPLE_RATE))
                kept.append(embed)
        return windows, (np.vstack(kept) if kept else np.empty((0, 0)))

    def _cluster(self, embeddings: np.ndarray, num_speakers: int | None) -> np.ndarray:
        from sklearn.cluster import AgglomerativeClustering

        if num_speakers is not None:
            n = max(1, min(num_speakers, len(embeddings)))
            model = AgglomerativeClustering(
                n_clusters=n, metric="cosine", linkage="average"
            )
            return model.fit_predict(embeddings)

        # Unknown cast size: let the threshold decide how many voices there
        # are, then collapse to max_speakers if it over-splits -- pipeline
        # treats an absurd count as a diarization failure and aborts the run,
        # so returning 30 "speakers" would be worse than returning 12.
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=self.distance_threshold,
            metric="cosine", linkage="average",
        )
        labels = model.fit_predict(embeddings)
        found = len(set(labels))
        if found > self.max_speakers:
            log.warning(
                "clustering found %d voices, collapsing to %d; pass --num-speakers "
                "for a known cast size", found, self.max_speakers,
            )
            model = AgglomerativeClustering(
                n_clusters=self.max_speakers, metric="cosine", linkage="average"
            )
            labels = model.fit_predict(embeddings)
        return labels

    def _speech_regions(self, speech_mask: np.ndarray) -> list[tuple[float, float]]:
        """Contiguous runs of speech, bridged across short pauses.

        These runs define what must be dubbed. Letting the embedding windows
        decide that instead dropped every stretch that failed the coverage or
        minimum-duration filters -- 36% of all dialogue on a real clip, which
        reached the finished dub as music playing over a talking character."""
        padded = np.concatenate(([False], speech_mask, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        runs = [
            (edges[i] / _SAMPLE_RATE, edges[i + 1] / _SAMPLE_RATE)
            for i in range(0, len(edges) - 1, 2)
        ]
        if not runs:
            return []

        # Bridge gaps shorter than this: a within-sentence breath is not a
        # turn boundary, and splitting on it produces slots too short to say
        # anything in.
        merged = [runs[0]]
        for start, end in runs[1:]:
            if start - merged[-1][1] <= _MAX_BRIDGED_PAUSE:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return [(a, b) for a, b in merged if b - a >= _MIN_TURN_DURATION]

    def _assign_to_speech_regions(
        self,
        speech_mask: np.ndarray,
        windows: list[tuple[float, float]],
        labels: np.ndarray,
        duration: float,
    ) -> list[Segment]:
        """Give every speech region a speaker, then split it where the
        speaker changes.

        Coverage comes from the VAD regions and identity from the clustered
        windows, so a region with no window of its own still gets dubbed --
        it inherits the nearest window's speaker rather than being dropped.
        """
        order: dict[int, str] = {}

        def speaker_id(label: int) -> str:
            if label not in order:
                order[label] = f"SPEAKER_{len(order):02d}"
            return order[label]

        centres = np.array([(a + b) / 2 for a, b in windows])
        segments: list[Segment] = []

        for region_start, region_end in self._speech_regions(speech_mask):
            inside = [
                i for i, (w_start, w_end) in enumerate(windows)
                if w_end > region_start and w_start < region_end
            ]
            if not inside:
                # No embedding window survived here, but somebody is still
                # talking: hand it to the nearest window's speaker instead of
                # leaving the line undubbed.
                nearest = int(np.argmin(np.abs(centres - (region_start + region_end) / 2)))
                segments.append(Segment(
                    start=region_start, end=region_end,
                    speaker_id=speaker_id(int(labels[nearest])),
                ))
                continue

            # Split the region where the speaker changes, at the midpoint
            # between the two windows that disagree.
            cuts: list[tuple[float, int]] = [(region_start, int(labels[inside[0]]))]
            for prev, cur in zip(inside, inside[1:]):
                if labels[cur] != labels[prev]:
                    cuts.append(((centres[prev] + centres[cur]) / 2, int(labels[cur])))
            cuts.append((region_end, -1))

            for (cut_start, label), (cut_end, _) in zip(cuts, cuts[1:]):
                if cut_end - cut_start >= _MIN_TURN_DURATION:
                    segments.append(Segment(
                        start=cut_start, end=cut_end, speaker_id=speaker_id(label),
                    ))

        segments.sort(key=lambda s: s.start)
        return segments
