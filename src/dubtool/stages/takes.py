"""Rejects a synthesized take that isn't speech.

CosyVoice2 samples every take, and now and then a take comes back with no
speech in it -- measured at 1 in 40 on real dialogue (a 1.6s "happy" take
of a two-word line with no voiced frames at all). Synthesis reports
success, the timing planner places it, and the line plays as noise or
silence: a line that seems to have been skipped. Nothing downstream can
tell, so the take is checked here and re-rolled.

The test is the fraction of voiced frames. Healthy takes on the same
material ran 58% voiced at the median and 41% at the 10th percentile; the
broken one was 0%. A whispered take is unvoiced by design, so for the
whispers preset only loudness is checked -- a voicing test would reject
every legitimate whisper.
"""
from __future__ import annotations

import librosa
import numpy as np

MIN_VOICED_FRACTION = 0.15
_MIN_RMS = 1e-3
_ANALYSIS_SR = 16000


def voiced_fraction(audio: np.ndarray, sr: int) -> float:
    if len(audio) == 0:
        return 0.0
    y = audio.astype(np.float32)
    if sr != _ANALYSIS_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=_ANALYSIS_SR)
    if len(y) < 2048:
        return 0.0
    _, voiced_flag, _ = librosa.pyin(y, fmin=65, fmax=1000, sr=_ANALYSIS_SR, frame_length=1024)
    return float(np.mean(voiced_flag)) if len(voiced_flag) else 0.0


def is_usable(audio: np.ndarray, sr: int, emotion: str | None) -> tuple[bool, float]:
    """(usable, voiced_fraction). voiced_fraction is reported as 1.0 for a
    whisper, which is exempt from the voicing test."""
    if len(audio) == 0 or float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) < _MIN_RMS:
        return False, 0.0
    if emotion == "whispers":
        return True, 1.0
    voiced = voiced_fraction(audio, sr)
    return voiced >= MIN_VOICED_FRACTION, voiced
