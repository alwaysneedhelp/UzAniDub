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

    result: dict[int, str] = {}
    for speaker_id, indices in by_speaker.items():
        pitches = [features[i][0] for i in indices if features[i][0] > 0]
        rmses = [features[i][2] for i in indices if features[i][2] > 0]
        if not pitches or not rmses:
            continue  # not enough signal to establish a baseline for this speaker
        baseline_pitch = float(np.median(pitches))
        baseline_rms = float(np.median(rmses))
        stds = [features[i][1] for i in indices if features[i][0] > 0 and features[i][1] > 0]
        baseline_pitch_std = float(np.median(stds)) if stds else 0.0

        for i in indices:
            pitch_mean, pitch_std, rms = features[i]
            if pitch_mean <= 0 or baseline_pitch <= 0 or baseline_rms <= 0:
                continue
            pitch_std_ratio = (pitch_std / baseline_pitch_std) if baseline_pitch_std > 0 else 1.0
            emotion = _classify(pitch_mean / baseline_pitch, rms / baseline_rms, pitch_std_ratio)
            if emotion:
                result[i] = emotion
                log.debug(
                    "segment %d (%s): pitch_ratio=%.2f energy_ratio=%.2f pitch_std_ratio=%.2f -> %s",
                    i, speaker_id, pitch_mean / baseline_pitch, rms / baseline_rms, pitch_std_ratio, emotion,
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


def _classify(pitch_ratio: float, energy_ratio: float, pitch_std_ratio: float = 1.0) -> str:
    """Always returns a preset — see the module docstring for why
    near-baseline delivery now maps to "calm" instead of returning None.
    Level-based thresholds (pitch_ratio/energy_ratio) are moderated from an
    earlier, stricter version that required a bigger deviation than most
    real dialogue actually reaches outside of genuinely extreme moments;
    checked before the variability-based nervous/tired branches so a
    strong level signal always wins over a variability one."""
    if energy_ratio < 0.55:
        return "whispers"
    if pitch_ratio > 1.4 and energy_ratio > 1.25:
        return "surprised"
    if pitch_ratio > 1.2 and energy_ratio > 1.1:
        return "happy"
    if energy_ratio > 1.3 and pitch_ratio <= 1.2:
        return "angry"
    if energy_ratio < 0.75 and pitch_ratio < 0.9:
        return "sad"
    if pitch_std_ratio > 1.6 and 0.8 <= energy_ratio <= 1.3:
        return "nervous"
    if pitch_std_ratio < 0.6 and 0.75 <= energy_ratio < 0.9:
        return "tired"
    return "calm"
