"""faster-whisper transcription with word-level timestamps."""
from __future__ import annotations

import logging
from pathlib import Path

from dubtool.types import Segment, Word

log = logging.getLogger(__name__)


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        language: str | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
        proper_nouns: list[str] | None = None,
    ):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._language = language  # None = auto-detect
        self._base_proper_nouns = list(proper_nouns) if proper_nouns else []

    def _initial_prompt(self, extra_proper_nouns: list[str] | None = None) -> str | None:
        # Whisper's initial_prompt biases decoding toward vocabulary it's
        # seen recently in the "context" — feeding known proper nouns here
        # measurably helps recognize names an ASR model would otherwise
        # guess at (e.g. transcribing an unfamiliar name as the nearest
        # common word/acronym it does know). Matters most for content like
        # anime with character names no general-purpose model has seen.
        names = list(self._base_proper_nouns)
        for name in extra_proper_nouns or []:
            if name.lower() not in (n.lower() for n in names):
                names.append(name)
        if not names:
            return None
        return "Character and place names in this audio: " + ", ".join(names) + "."

    def transcribe(
        self, audio_path: Path, segments: list[Segment], extra_proper_nouns: list[str] | None = None
    ) -> list[Segment]:
        """`extra_proper_nouns` adds to (not replaces) whatever proper nouns
        this transcriber was constructed with — used for a second
        auto-names-informed transcription pass (see pipeline.py) without
        needing a second transcriber instance."""
        whisper_segments, info = self._model.transcribe(
            str(audio_path),
            language=self._language,
            word_timestamps=True,
            initial_prompt=self._initial_prompt(extra_proper_nouns),
            # Whisper's default (True) feeds each segment's own transcript
            # back in as context for decoding the next one — this is the
            # documented, common cause of hallucination/repetition loops on
            # longer files: one bad decode poisons the context and the
            # model gets stuck echoing an earlier line instead of
            # transcribing what's actually being said, sometimes for
            # minutes at a stretch. Diagnosed on real content: a 5-minute
            # documentary produced the exact same sentence verbatim at four
            # wildly different timestamps (158s/218s/249s/279s) while the
            # real, different narration at those points (confirmed by
            # re-transcribing those exact spans in isolation, where it came
            # out correctly) was silently lost.
            condition_on_previous_text=False,
        )

        result: list[Segment] = []
        for ws in whisper_segments:
            words = [Word(text=w.word.strip(), start=w.start, end=w.end) for w in (ws.words or [])]
            result.append(
                Segment(
                    start=ws.start,
                    end=ws.end,
                    speaker_id=_speaker_at(segments, (ws.start + ws.end) / 2),
                    text=ws.text.strip(),
                    words=words,
                    detected_language=info.language,
                )
            )
        log.info(
            "transcribed %d segment(s), detected language=%s (p=%.2f)",
            len(result), info.language, info.language_probability,
        )
        return result


def _speaker_at(diarized_segments: list[Segment], t: float) -> str:
    """Which diarized speaker was talking at time `t`.

    Whisper transcribes the whole track in one pass, independent of
    diarization boundaries — its own segments don't inherit a speaker_id for
    free, they have to be matched back against the diarized turns by time.
    A single-speaker diarizer makes this a no-op (everything matches the one
    turn), which is exactly why an earlier version of this function got away
    with just grabbing `segments[0].speaker_id` for everything — that silently
    breaks the moment a real multi-speaker diarizer is used, collapsing every
    speaker's lines onto whichever one happened to be diarized first.
    """
    for seg in diarized_segments:
        if seg.start <= t < seg.end:
            return seg.speaker_id
    if not diarized_segments:
        return "SPEAKER_00"
    return min(diarized_segments, key=lambda s: min(abs(s.start - t), abs(s.end - t))).speaker_id
