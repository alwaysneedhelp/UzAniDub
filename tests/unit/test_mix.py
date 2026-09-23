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


def test_clips_are_scaled_down_to_avoid_overflow(tmp_path):
    sr = 1000
    background = np.full(int(1.0 * sr), 0.9, dtype=np.float32)
    background_path = _write_wav(tmp_path / "background.wav", background, sr)

    seg = Segment(start=0.0, end=1.0, speaker_id="SPEAKER_00")
    seg.synthesized_audio = np.full(int(1.0 * sr), 0.9, dtype=np.float32)  # 0.9+0.9=1.8, would clip

    out_path = mix_segments([seg], background_path, tmp_path / "final.wav", sample_rate=sr)
    mixed, _ = sf.read(str(out_path))
    assert np.abs(mixed).max() <= 0.98 + 1e-6
