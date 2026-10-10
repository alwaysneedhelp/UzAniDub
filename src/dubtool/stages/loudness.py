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


# --- Following the actor -------------------------------------------------
#
# normalize() above puts every line on one of a few fixed levels, all within
# about 2dB of each other. Measured on a real clip, that squeezed the actor's
# 9.8dB line-to-line range into 2.3dB: a shout sat 5.7dB above an ordinary
# line in the original and 1.5dB above it in the dub. That contrast between
# lines is a large part of what makes a scene sound emotional rather than
# read out. So each line now takes the loudness the actor gave that line,
# measured on the separated vocals -- which also puts the dubbed voice where
# the original dialogue sat against the music.

# Loudness is compared over active speech only -- frames within this many dB
# of the line's loudest -- so pauses inside a line don't drag its level down.
_ACTIVE_FLOOR_DB = 30.0
_FRAME_SECONDS = 0.02
# How far a single line may sit from the scene's typical line: about -9dB to
# +9.5dB, wide enough for a whisper or a shout, narrow enough that a line whose
# window caught a burst of effects or a missed boundary can't go wild.
_MIN_RATIO_TO_SCENE = 0.35
_MAX_RATIO_TO_SCENE = 3.0
_PEAK_CEILING = 0.95


def active_rms(audio: np.ndarray, sr: int) -> float:
    frame = max(1, int(_FRAME_SECONDS * sr))
    n = len(audio) // frame
    if n == 0:
        return 0.0
    frames = np.sqrt(np.mean(audio[: n * frame].astype(np.float64).reshape(n, frame) ** 2, axis=1))
    loudest = float(frames.max())
    if loudest <= 0:
        return 0.0
    active = frames[frames >= loudest * 10 ** (-_ACTIVE_FLOOR_DB / 20)]
    return float(np.sqrt(np.mean(active**2)))


def actor_targets(segments, vocals_path) -> list[float | None]:
    """Per line, the active speech level the actor gave it, clamped to a sane
    distance from the scene's median line. None where there's nothing to
    measure; the caller falls back to normalize() for those."""
    import soundfile as sf

    data, sr = sf.read(str(vocals_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    levels = [
        active_rms(data[max(0, int(s.start * sr)):min(len(data), int(s.end * sr))], sr)
        for s in segments
    ]
    measured = [lv for lv in levels if lv > 1e-5]
    if not measured:
        return [None] * len(segments)
    scene = float(np.median(measured))
    return [
        float(np.clip(lv, scene * _MIN_RATIO_TO_SCENE, scene * _MAX_RATIO_TO_SCENE)) if lv > 1e-5 else None
        for lv in levels
    ]


def match_level(audio: np.ndarray, sr: int, target_active_rms: float) -> np.ndarray:
    """Scale a line so its active speech level equals `target_active_rms`
    (never by more than _MAX_GAIN), then peak-limit it.

    Capping the *gain* at ceiling/peak instead held back exactly the lines
    that need the most level: a shouted line at its target has peaks well
    past full scale, so one transient pulled the whole line down -- the
    scene's shout came out +1.1dB over an ordinary line instead of the
    actor's +5.5dB. A limiter turns down only the moments that would clip.
    """
    current = active_rms(audio, sr)
    if current <= 1e-6 or len(audio) == 0:
        return audio
    gain = min(target_active_rms / current, _MAX_GAIN)
    out = _limit((audio * gain).astype(np.float32), sr, _PEAK_CEILING)
    # Limiting a loud line costs it level: voiced speech peaks every pitch
    # period, so keeping a shout's peaks under the ceiling turns its body
    # down too. Make that back up -- which compresses the line, the way
    # broadcast dialogue is -- but by at most _MAX_MAKEUP_DB, so ordinary
    # lines are never squashed to hit a number.
    budget = 10 ** (_MAX_MAKEUP_DB / 20)
    for _ in range(_MAKEUP_ROUNDS):
        reached = active_rms(out, sr)
        if reached <= 1e-6 or reached >= target_active_rms * 0.97:
            break
        step = min(target_active_rms / reached, budget)
        budget /= step
        gain = min(gain * step, _MAX_GAIN * 10 ** (_MAX_MAKEUP_DB / 20))
        out = _limit((audio * gain).astype(np.float32), sr, _PEAK_CEILING)
        if budget <= 1.0001:
            break
    return out


_MAX_MAKEUP_DB = 6.0
_MAKEUP_ROUNDS = 4


_LIMIT_WINDOW_SECONDS = 0.008


def _limit(audio: np.ndarray, sr: int, ceiling: float) -> np.ndarray:
    """Look-ahead peak limiter: no sample leaves above `ceiling`.

    The gain each sample needs is minimum-filtered over +/-8ms and then
    averaged over the same span. Every value in that average is at most
    the need at the sample being protected, so the guarantee holds while the
    gain change is spread over ~16ms instead of applied sample by sample --
    turning peaks down cleanly rather than waveshaping them."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak <= ceiling:
        return audio
    need = np.minimum(1.0, ceiling / np.maximum(np.abs(audio), 1e-12))
    window = 2 * max(1, int(_LIMIT_WINDOW_SECONDS * sr)) + 1
    gain = uniform_filter1d(minimum_filter1d(need, size=window, mode="nearest"), size=window, mode="nearest")
    return (audio * np.minimum(gain, need)).astype(np.float32)


# --- Vocal effort ----------------------------------------------------------
#
# A scream is not only louder: effort pushes a voice's energy into the
# presence range (roughly 2-4kHz), which is where its raw, strained edge
# lives. The TTS doesn't do this on its own. Measured on a screamed line,
# every instruction -- stock presets and purpose-written "scream at the top
# of your lungs" ones alike -- landed at the same ~1550Hz spectral centroid;
# the strongest preset (angry) already gave +6.5 semitones of pitch and
# custom wording added nothing past it. So the effort is added afterwards,
# in proportion to how hard the actor delivered the line.

# Lines delivered this many dB over the scene's median line start to get
# effort; by the second figure they get all of it. An ordinary exclamation
# sits around +2.5dB; the screamed line on the test clip at +5.5dB.
_EFFORT_START_DB = 2.5
_EFFORT_FULL_DB = 6.0
_MAX_EFFORT_DB = 6.0
_PRESENCE_HZ = 2800.0
_PRESENCE_Q = 0.7


def effort_for(line_db_over_scene: float) -> float:
    """dB of presence boost for a line the actor delivered this many dB
    above the scene's median line: none for ordinary delivery, rising
    linearly to _MAX_EFFORT_DB for a full scream."""
    span = _EFFORT_FULL_DB - _EFFORT_START_DB
    return float(np.clip((line_db_over_scene - _EFFORT_START_DB) / span, 0.0, 1.0) * _MAX_EFFORT_DB)


def vocal_effort(audio: np.ndarray, sr: int, boost_db: float) -> np.ndarray:
    """A broad presence boost (RBJ peaking EQ) -- timbre only; the caller
    levels the line afterwards, so this changes how it sounds, not how loud
    it is."""
    if boost_db <= 0.05 or len(audio) == 0 or _PRESENCE_HZ >= sr / 2:
        return audio
    from scipy.signal import lfilter

    a_gain = 10 ** (boost_db / 40)
    w0 = 2 * np.pi * _PRESENCE_HZ / sr
    alpha = np.sin(w0) / (2 * _PRESENCE_Q)
    b = np.array([1 + alpha * a_gain, -2 * np.cos(w0), 1 - alpha * a_gain])
    a = np.array([1 + alpha / a_gain, -2 * np.cos(w0), 1 - alpha / a_gain])
    return lfilter(b / a[0], a / a[0], audio).astype(np.float32)
