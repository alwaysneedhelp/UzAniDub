"""Time-aligns synthesized speech to the source segment's duration.

Translated Uzbek text is very unlikely to take exactly as long to speak as
the original language did. We time-stretch/compress the synthesized audio
with a phase vocoder (librosa) to close that gap.

This is a real tradeoff, not just an implementation detail: push the stretch
ratio too far and speech starts sounding unnatural (phase-vocoder pitch/
timbre artifacts, or unnaturally fast/slow pacing) well before it sounds
*wrong* in a lip-sync sense. `min_stretch_ratio`/`max_stretch_ratio` cap how
far we'll push it — outside that range we accept timing drift (the dubbed
segment starts on time but may end early/late relative to the original)
rather than degrade the voice further. There's no universally correct
setting; tighten the bounds for lip-sync-critical content, loosen them for
narration where naturalness matters more than sync.
"""
from __future__ import annotations

import logging

import librosa
import numpy as np

log = logging.getLogger(__name__)


def align_segment(
    audio: np.ndarray,
    sr: int,
    target_duration: float,
    min_ratio: float = 0.5,
    max_ratio: float = 2.0,
) -> np.ndarray:
    """`stretch_ratio` = target_duration / current_duration: >1 means we're
    lengthening (slowing speech down), <1 means shortening (speeding up).
    `min_ratio`/`max_ratio` bound how far this function will stretch audio.
    """
    if len(audio) == 0:
        return audio
    current_duration = len(audio) / sr
    if current_duration <= 0 or target_duration <= 0:
        return audio

    stretch_ratio = target_duration / current_duration
    clamped_ratio = max(min_ratio, min(max_ratio, stretch_ratio))
    if clamped_ratio != stretch_ratio:
        log.warning(
            "segment wants stretch ratio %.2fx (target=%.2fs, synthesized=%.2fs); "
            "clamping to %.2fx and accepting timing drift",
            stretch_ratio, target_duration, current_duration, clamped_ratio,
        )

    if abs(clamped_ratio - 1.0) < 0.02:
        return audio  # close enough; skip resampling artifacts for no benefit

    rate = 1.0 / clamped_ratio  # librosa's `rate` speeds up; we want to apply `stretch_ratio`
    return librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)
