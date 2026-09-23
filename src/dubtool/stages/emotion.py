"""Classifies each segment's likely emotional delivery from its own
original audio (pitch/energy relative to the speaker's own baseline
elsewhere in the file), for routing into CosyVoice2's inference_instruct2
emotion presets (see backends/tts_cosyvoice.py) instead of leaving delivery
purely to zero-shot's implicit reference-clip transfer.

Deliberately audio-based, not text-based: a text emotion classifier would
only work reliably for English source content (breaking the "any source
language" design), and the readily-available pretrained options checked
during research didn't have a clearly stated license, which this project
treats as a hard requirement (see THIRD_PARTY_LICENSES.md). Pitch/energy
analysis is plain math over our own Apache-2.0 code, works for any spoken
language, and was already validated as a real signal — this reuses the
same pitch-analysis finding that motivated stages/reference.py's
dynamic-range-aware clip scoring.

This only overrides natural prosody with an explicit instruction when a
segment's acoustic deviation from *its own speaker's baseline* is strong
enough to be a confident signal; ambiguous/near-baseline segments are left
out of the result entirely, so the caller keeps using zero-shot (natural
reference-clip prosody) rather than risking a wrong classification
actively degrading a segment that was already delivering fine on its own.
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
    """Returns {segment_index: emotion_preset_name} for segments whose
    acoustic profile deviates strongly enough from their speaker's own
    baseline to warrant an explicit emotion instruction. Segment indices
    absent from the returned dict should keep using natural prosody.

    Preset names match aisha-org/navoiy-tts's emotions_40h.json tags
    (calm, happy, sad, angry, nervous, surprised, whispers, warm, tired,
    sarcastic) — only a subset are reachable from acoustic features alone
    (sarcasm and warmth aren't acoustically distinctive the way loudness or
    pitch spikes are), so this only ever returns from that reachable subset.
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

        for i in indices:
            pitch_mean, _pitch_std, rms = features[i]
            if pitch_mean <= 0 or baseline_pitch <= 0 or baseline_rms <= 0:
                continue
            emotion = _classify(pitch_mean / baseline_pitch, rms / baseline_rms)
            if emotion:
                result[i] = emotion
                log.debug(
                    "segment %d (%s): pitch_ratio=%.2f energy_ratio=%.2f -> %s",
                    i, speaker_id, pitch_mean / baseline_pitch, rms / baseline_rms, emotion,
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


def _classify(pitch_ratio: float, energy_ratio: float) -> str | None:
    if energy_ratio < 0.5:
        return "whispers"
    if pitch_ratio > 1.5 and energy_ratio > 1.3:
        return "surprised"
    if pitch_ratio > 1.3 and energy_ratio > 1.2:
        return "happy"
    if energy_ratio > 1.4 and pitch_ratio <= 1.3:
        return "angry"
    if energy_ratio < 0.65 and pitch_ratio < 0.85:
        return "sad"
    return None  # ambiguous/near-baseline -- let zero-shot handle it
