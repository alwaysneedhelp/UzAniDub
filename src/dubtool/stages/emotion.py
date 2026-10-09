"""Classifies each segment's likely emotional delivery from its own
original audio (pitch/energy relative to the speaker's own baseline
elsewhere in the file), for routing into CosyVoice2's inference_instruct2
emotion presets (see backends/tts_cosyvoice.py).

Deliberately audio-based, not text-based: a text emotion classifier would
only work reliably for English source content (breaking the "any source
language" design), and the readily-available pretrained options checked
during research didn't have a clearly stated license, which this project
treats as a hard requirement (see THIRD_PARTY_LICENSES.md). Pitch/energy
analysis is plain math over our own Apache-2.0 code, works for any spoken
language, and was already validated as a real signal — this reuses the
same pitch-analysis finding that motivated stages/reference.py's
dynamic-range-aware clip scoring.

Earlier versions of this function only returned a classification for
acoustically extreme segments, leaving near-baseline ones out of the
result entirely so the caller fell back to zero-shot (cloning the
reference clip's own prosody directly, no explicit style instruction).
That turned out to be backwards: real testing kept coming back "flat" or
"robotic" even with a per-segment reference clip cloned from real,
expressive source audio — the navoiy-tts checkpoint was fine-tuned mainly
on a set of discrete named emotion presets (aisha-org/navoiy-tts's
emotions_40h.json), so *zero-shot* mode — which never got that same
fine-tuning attention — is the actual weak link, not voice cloning
itself. This now classifies (almost) every segment into its
closest-matching preset instead, defaulting near-baseline delivery to
"calm" rather than leaving it unset, so instruct2 (which the checkpoint
demonstrably handles well) is used for nearly everything; the reference
clip passed alongside it still supplies voice *identity* (see
backends/tts_cosyvoice.py — instruct2 takes reference_audio too), so this
doesn't trade away cloning, it trades away the zero-shot delivery path
specifically. A segment is left unclassified only when there's no usable
pitch signal at all (near-silent, unvoiced, or too short to track).

Loudness is now judged against the *whole scene*, not each speaker's own
median, and the line's punctuation counts. Both changes come from reading
the classifier's inputs on real dubbed output:

- Per-speaker baselines made a character who shouts throughout "calm":
  their median line *is* a shout. And a speaker diarization gives only
  one or two lines is mostly their own baseline, so everything they say
  lands near 1.0. In a professionally mixed dub dialogue is levelled
  across characters, so absolute loudness within the scene is the honest
  signal -- shouting is loud, a mutter is quiet, whoever says it.
- "!" and "?" from the transcript are a delivery cue the audio features
  missed: "Are you for real?" was classified as a whisper for being quiet,
  "I could never do that till now!" as calm. This is a character check,
  not a text classifier -- it needs no model and no license, and the
  transcriber emits punctuation for every language it handles.
- Pitch *variability* no longer selects "nervous" or "tired". It fired on
  an animated explanation as easily as on nerves, and turned plain
  declaratives into a slow, drained "tired" reading. Both presets remain
  available through --emotion.

Measured alongside this: navoiy's presets do not reliably separate in
pitch range or loudness variation (mean range 7.7-9.9 semitones across
calm/happy/angry/surprised, vs the source actor's 6.8), and zero-shot was
no better and far less predictable. So the leverage is in not sending an
obviously wrong preset -- a whisper, a sorrowful reading -- for a line
whose delivery is plain from its loudness and punctuation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from dubtool.types import Segment

log = logging.getLogger(__name__)


def classify_segments(segments: list[Segment], vocals_path: Path) -> dict[int, str]:
    """Returns {segment_index: emotion_preset_name} for (almost) every
    segment — see the module docstring for why this now classifies nearly
    everything instead of only acoustically extreme segments. A segment
    index is absent from the result only when there was no usable pitch
    signal for it at all (near-silent, unvoiced, or too short to track),
    in which case the caller should fall back to zero-shot.

    Preset names match aisha-org/navoiy-tts's emotions_40h.json tags
    (calm, happy, sad, angry, nervous, surprised, whispers, warm, tired,
    sarcastic) — only a subset are reachable from acoustic features alone
    (sarcasm and warmth aren't acoustically distinctive the way loudness or
    pitch spikes are), so this only ever returns from that reachable subset.
    "nervous" and "tired" are reached via pitch *variability* (how much
    pitch wavers within the segment, relative to the speaker's own
    baseline variability) rather than pitch/energy level — a shaky, uneven
    pitch contour at otherwise-ordinary loudness reads as nervous; an
    unusually flat, monotone contour at slightly-low energy reads as
    tired, distinct from "sad" (which requires both low energy *and* a
    lower-than-usual pitch, not just reduced variability).
    """
    data, sr = sf.read(str(vocals_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)

    by_speaker: dict[str, list[int]] = {}
    for i, seg in enumerate(segments):
        by_speaker.setdefault(seg.speaker_id, []).append(i)

    features: dict[int, tuple[float, float, float]] = {}
    for i, seg in enumerate(segments):
        start = max(0, int(seg.start * sr))
        end = min(len(data), int(seg.end * sr))
        features[i] = _pitch_and_energy(data[start:end], sr)

    rmses = [features[i][2] for i in features if features[i][2] > 0]
    if not rmses:
        return {}
    scene_rms = float(np.median(rmses))

    result: dict[int, str] = {}
    for speaker_id, indices in by_speaker.items():
        pitches = [features[i][0] for i in indices if features[i][0] > 0]
        # A pitch baseline needs a few lines to mean anything; with fewer,
        # a speaker's only line is its own baseline and every ratio is 1.0.
        baseline_pitch = float(np.median(pitches)) if len(pitches) >= _MIN_LINES_FOR_PITCH else 0.0

        for i in indices:
            pitch_mean, _, rms = features[i]
            # Skip only real silence. A line pitch tracking couldn't follow
            # (shouted over music, say) still has a loudness and still has
            # its punctuation -- "Exactly!" was left unclassified this way.
            if rms <= _SILENCE_RMS or scene_rms <= 0:
                continue
            pitch_ratio = (
                pitch_mean / baseline_pitch if pitch_mean > 0 and baseline_pitch > 0 else None
            )
            emotion = _classify(rms / scene_rms, pitch_ratio, segments[i].text)
            result[i] = emotion
            log.debug(
                "segment %d (%s): energy_vs_scene=%.2f pitch_ratio=%s punct=%r -> %s",
                i, speaker_id, rms / scene_rms,
                f"{pitch_ratio:.2f}" if pitch_ratio is not None else "n/a",
                _punctuation(segments[i].text), emotion,
            )
    return result


def _pitch_and_energy(data: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Returns (pitch_mean, pitch_std, rms); pitch values are 0 for a clip
    too short/silent/unvoiced for reliable pitch tracking."""
    rms = float(np.sqrt(np.mean(data**2))) if len(data) else 0.0
    if len(data) < sr * 0.2:
        return 0.0, 0.0, rms
    try:
        f0, voiced_flag, _ = librosa.pyin(
            data, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
    except Exception:
        return 0.0, 0.0, rms
    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) == 0:
        return 0.0, 0.0, rms
    return float(np.mean(voiced_f0)), float(np.std(voiced_f0)), rms


_MIN_LINES_FOR_PITCH = 3
_SILENCE_RMS = 1e-4
_EXCLAIM = "!！¡"
_QUESTION = "?？¿"
# A question this short is a reaction ("Are you for real?", "Is he high?",
# "What?"), not a request for information, and reads as surprise.
_REACTION_MAX_WORDS = 5


def _punctuation(text: str) -> str:
    return ("!" if any(c in text for c in _EXCLAIM) else "") + (
        "?" if any(c in text for c in _QUESTION) else ""
    )


def _classify(energy_vs_scene: float, pitch_ratio: float | None, text: str = "") -> str:
    """Always returns a preset. `energy_vs_scene` is the line's loudness over
    the scene's median line; `pitch_ratio` is against the speaker's own
    median pitch, or None when the speaker has too few lines to have one.

    Order matters: a shout is a shout whatever its punctuation, and
    punctuation outranks the quieter-delivery presets, so an exclamation or
    a reaction is never read as a whisper or as sorrow."""
    punct = _punctuation(text)
    exclaim, question = "!" in punct, "?" in punct
    words = len(text.split())

    if energy_vs_scene > 1.7:
        # A shout. Anger raises pitch moderately; a jump far past the
        # speaker's usual pitch on top of it is a shriek -- alarm, surprise.
        if pitch_ratio is not None and pitch_ratio > 1.4:
            return "surprised"
        return "angry"
    if exclaim:
        return "happy"            # navoiy's "happy, excited and energetic" preset
    if question and (energy_vs_scene >= 0.9 or words <= _REACTION_MAX_WORDS):
        return "surprised"
    if energy_vs_scene < 0.45 and not question:
        return "whispers"
    if energy_vs_scene > 1.35:
        return "angry"            # forceful without being marked as an exclamation
    if energy_vs_scene < 0.6 and pitch_ratio is not None and pitch_ratio < 0.85:
        return "sad"
    return "calm"
