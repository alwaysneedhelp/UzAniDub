import numpy as np

from dubtool.stages.align import align_segment


def _tone(duration_s: float, sr: int = 24000) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    return np.sin(2 * np.pi * 220 * t).astype(np.float32)


def test_stretches_to_target_duration_within_bounds():
    sr = 24000
    audio = _tone(1.0, sr)  # 1s synthesized
    out = align_segment(audio, sr, target_duration=1.5, min_ratio=0.5, max_ratio=2.0)
    assert abs(len(out) / sr - 1.5) < 0.05


def test_compresses_to_target_duration_within_bounds():
    sr = 24000
    audio = _tone(2.0, sr)
    out = align_segment(audio, sr, target_duration=1.0, min_ratio=0.5, max_ratio=2.0)
    assert abs(len(out) / sr - 1.0) < 0.05


def test_clamps_extreme_stretch_and_does_not_crash():
    sr = 24000
    audio = _tone(1.0, sr)
    # wants a 10x stretch; should clamp to max_ratio=2.0 rather than blow up
    out = align_segment(audio, sr, target_duration=10.0, min_ratio=0.5, max_ratio=2.0)
    assert abs(len(out) / sr - 2.0) < 0.05


def test_skips_processing_when_already_close_enough():
    sr = 24000
    audio = _tone(1.0, sr)
    out = align_segment(audio, sr, target_duration=1.005, min_ratio=0.5, max_ratio=2.0)
    assert np.array_equal(out, audio)


def test_empty_audio_passthrough():
    out = align_segment(np.array([], dtype=np.float32), 24000, target_duration=1.0)
    assert len(out) == 0
