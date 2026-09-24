"""Normalizes each synthesized segment to a consistent target loudness
before mixing.

CosyVoice2 renders each segment independently, and its output level varies
noticeably from line to line even for ostensibly similar delivery — with no
normalization, that shows up as audibly inconsistent volume from one dubbed
line to the next, unrelated to anything about the actual content (verified
as a real, reported issue on real output, not a theoretical concern).

Target levels are emotion-aware rather than one flat number: forcing an
segment classified "whispers" (see stages/emotion.py) up to the same level
as everything else would defeat the point of classifying it as a whisper in
the first place. Segments with no classified emotion — the majority, left
on natural zero-shot prosody — all share one baseline target, which is
where the actual inconsistency showed up in practice: there's no legitimate
reason for those to vary in overall loudness from one to the next.
"""
from __future__ import annotations

import numpy as np

_DEFAULT_TARGET_RMS = 0.1
_EMOTION_TARGET_RMS = {
    "whispers": 0.035,
    "sad": 0.07,
    "tired": 0.07,
    "angry": 0.13,
    "surprised": 0.13,
    "happy": 0.12,
}
# Caps amplification so a near-silent/failed synthesis doesn't get blown up
# into audible noise trying to hit the target.
_MAX_GAIN = 6.0


def normalize(audio: np.ndarray, emotion: str | None = None) -> np.ndarray:
    if len(audio) == 0:
        return audio
    current_rms = float(np.sqrt(np.mean(audio**2)))
    if current_rms <= 1e-6:
        return audio  # silence/near-silence — nothing sensible to normalize toward
    target = _EMOTION_TARGET_RMS.get(emotion or "", _DEFAULT_TARGET_RMS)
    gain = min(target / current_rms, _MAX_GAIN)
    return audio * gain
