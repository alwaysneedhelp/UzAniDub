from pathlib import Path

import numpy as np
import soundfile as sf

from dubtool.stages.mix import mix_segments
from dubtool.types import Segment


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> Path:
    sf.write(str(path), audio, sr)
    return path


def test_places_segments_at_correct_offsets_and_sums_background(tmp_path):
    sr = 1000  # low rate keeps this test fast and the math easy to check
    background = np.full(int(2.0 * sr), 0.1, dtype=np.float32)
    background_path = _write_wav(tmp_path / "background.wav", background, sr)

    seg = Segment(start=0.5, end=1.0, speaker_id="SPEAKER_00")
    seg.synthesized_audio = np.full(int(0.5 * sr), 0.2, dtype=np.float32)

    out_path = mix_segments([seg], background_path, tmp_path / "final.wav", sample_rate=sr)
    mixed, out_sr = sf.read(str(out_path))

    assert out_sr == sr
    assert len(mixed) == len(background)
    # before the segment: background only
    assert np.allclose(mixed[: int(0.4 * sr)], 0.1, atol=1e-4)
    # during the segment: background + synthesized
    assert np.allclose(mixed[int(0.6 * sr) : int(0.9 * sr)], 0.3, atol=1e-4)
    # after the segment: background only again
    assert np.allclose(mixed[int(1.1 * sr) :], 0.1, atol=1e-4)


def test_extends_timeline_if_segment_runs_past_background(tmp_path):
    sr = 1000
    background = np.zeros(int(1.0 * sr), dtype=np.float32)
    background_path = _write_wav(tmp_path / "background.wav", background, sr)

    seg = Segment(start=0.8, end=1.5, speaker_id="SPEAKER_00")
    seg.synthesized_audio = np.full(int(0.7 * sr), 0.5, dtype=np.float32)

    out_path = mix_segments([seg], background_path, tmp_path / "final.wav", sample_rate=sr)
    mixed, _ = sf.read(str(out_path))
    assert len(mixed) >= int(1.5 * sr)


def test_overrunning_segment_is_truncated_before_the_next_segment_starts(tmp_path):
    # Timing correction sometimes lets a segment's synthesized audio run
    # longer than its original slot (see align.py/config.py) rather than
    # distort the voice further -- without a cap, that overrun would sum
    # directly on top of the next segment's own audio instead of just
    # drifting out of sync.
    sr = 1000
    background = np.zeros(int(3.0 * sr), dtype=np.float32)
    background_path = _write_wav(tmp_path / "background.wav", background, sr)

    seg_a = Segment(start=0.0, end=1.0, speaker_id="A")
    seg_a.synthesized_audio = np.full(int(1.5 * sr), 0.5, dtype=np.float32)  # runs 0.5s past its slot
    seg_b = Segment(start=1.2, end=2.0, speaker_id="B")
    seg_b.synthesized_audio = np.full(int(0.5 * sr), 0.3, dtype=np.float32)

    out_path = mix_segments([seg_a, seg_b], background_path, tmp_path / "final.wav", sample_rate=sr)
    mixed, _ = sf.read(str(out_path))

    # seg_a's overrun must not still be playing once seg_b starts
    assert np.allclose(mixed[int(1.25 * sr) : int(1.4 * sr)], 0.3, atol=1e-2)
    # seg_b plays at its own, undistorted level -- no leftover overlap from seg_a
    assert float(np.abs(mixed[int(1.3 * sr)])) < 0.9


def test_segment_fully_consumed_by_the_next_ones_start_is_skipped_without_erroring(tmp_path):
    sr = 1000
    background = np.zeros(int(2.0 * sr), dtype=np.float32)
    background_path = _write_wav(tmp_path / "background.wav", background, sr)

    seg_a = Segment(start=0.0, end=1.0, speaker_id="A")
    seg_a.synthesized_audio = np.full(int(1.0 * sr), 0.5, dtype=np.float32)
    seg_b = Segment(start=0.0, end=0.5, speaker_id="B")  # degenerate: same start as seg_a
    seg_b.synthesized_audio = np.full(int(0.2 * sr), 0.3, dtype=np.float32)

    # should not raise even though one segment's slot is immediately
    # consumed by a same-start neighbor (a zero-length "keep")
    mix_segments([seg_b, seg_a], background_path, tmp_path / "final.wav", sample_rate=sr)


def test_clips_are_scaled_down_to_avoid_overflow(tmp_path):
    sr = 1000
    background = np.full(int(1.0 * sr), 0.9, dtype=np.float32)
    background_path = _write_wav(tmp_path / "background.wav", background, sr)

    seg = Segment(start=0.0, end=1.0, speaker_id="SPEAKER_00")
    seg.synthesized_audio = np.full(int(1.0 * sr), 0.9, dtype=np.float32)  # 0.9+0.9=1.8, would clip

    out_path = mix_segments([seg], background_path, tmp_path / "final.wav", sample_rate=sr)
    mixed, _ = sf.read(str(out_path))
    assert np.abs(mixed).max() <= 0.98 + 1e-6
