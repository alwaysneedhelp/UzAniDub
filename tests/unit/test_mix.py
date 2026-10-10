from pathlib import Path

import numpy as np
import pytest
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


def test_a_line_that_runs_long_plays_in_full_and_the_next_waits(tmp_path):
    # Cutting a line off at the next line's start is what made dubbed
    # dialogue sound like it skipped words: Uzbek runs longer than the
    # English it replaces, and 13 of 16 lines on a real clip were cut. Now
    # the long line finishes and the next one starts after it instead.
    sr = 1000
    background = np.zeros(int(4.0 * sr), dtype=np.float32)
    background_path = _write_wav(tmp_path / "background.wav", background, sr)

    seg_a = Segment(start=0.0, end=1.0, speaker_id="A")
    seg_a.synthesized_audio = np.full(int(1.5 * sr), 0.5, dtype=np.float32)  # 0.5s past its slot
    seg_b = Segment(start=1.2, end=2.0, speaker_id="B")
    seg_b.synthesized_audio = np.full(int(0.5 * sr), 0.3, dtype=np.float32)

    out_path = mix_segments([seg_a, seg_b], background_path, tmp_path / "final.wav",
                            sample_rate=sr, min_gap=0.1)
    mixed, _ = sf.read(str(out_path))

    # all of seg_a is heard, including the overrun past where seg_b was due
    assert np.allclose(mixed[int(1.3 * sr):int(1.45 * sr)], 0.5, atol=1e-4)
    # seg_b waits for it (1.5s + 0.1s gap) and is heard in full, never summed on top
    assert np.allclose(mixed[int(1.62 * sr):int(2.08 * sr)], 0.3, atol=1e-4)
    assert float(np.abs(mixed).max()) <= 0.5 + 1e-4


def test_two_lines_due_at_the_same_moment_are_both_heard_in_turn(tmp_path):
    sr = 1000
    background = np.zeros(int(3.0 * sr), dtype=np.float32)
    background_path = _write_wav(tmp_path / "background.wav", background, sr)

    seg_a = Segment(start=0.0, end=1.0, speaker_id="A")
    seg_a.synthesized_audio = np.full(int(1.0 * sr), 0.5, dtype=np.float32)
    seg_b = Segment(start=0.0, end=0.5, speaker_id="B")
    seg_b.synthesized_audio = np.full(int(0.2 * sr), 0.3, dtype=np.float32)

    out_path = mix_segments([seg_b, seg_a], background_path, tmp_path / "final.wav",
                            sample_rate=sr, min_gap=0.0)
    mixed, _ = sf.read(str(out_path))
    # neither line is dropped and they never overlap
    assert np.isclose(mixed[int(0.5 * sr)], 0.5, atol=1e-4) or np.isclose(mixed[int(0.5 * sr)], 0.3, atol=1e-4)
    assert float(np.abs(mixed).max()) <= 0.5 + 1e-4
    assert np.count_nonzero(np.isclose(mixed, 0.3, atol=1e-4)) == int(0.2 * sr)


def test_placed_start_from_the_planner_is_honoured(tmp_path):
    sr = 1000
    background = np.zeros(int(3.0 * sr), dtype=np.float32)
    background_path = _write_wav(tmp_path / "background.wav", background, sr)

    seg = Segment(start=0.5, end=1.0, speaker_id="A")
    seg.synthesized_audio = np.full(int(0.5 * sr), 0.4, dtype=np.float32)
    seg.placed_start = 0.9  # the planner decided this line waits

    out_path = mix_segments([seg], background_path, tmp_path / "final.wav", sample_rate=sr)
    mixed, _ = sf.read(str(out_path))
    assert np.allclose(mixed[int(0.5 * sr):int(0.85 * sr)], 0.0, atol=1e-4)
    assert np.allclose(mixed[int(0.95 * sr):int(1.35 * sr)], 0.4, atol=1e-4)


def test_stereo_background_stays_stereo_with_the_voice_centred(tmp_path):
    # The background is the original music and effects; it used to be
    # downmixed to mono before the voice went on top.
    sr = 1000
    background = np.zeros((int(2.0 * sr), 2), dtype=np.float32)
    background[:, 0] = 0.1   # left
    background[:, 1] = -0.1  # right
    background_path = _write_wav(tmp_path / "background.wav", background, sr)

    seg = Segment(start=0.5, end=1.0, speaker_id="A")
    seg.synthesized_audio = np.full(int(0.5 * sr), 0.2, dtype=np.float32)

    out_path = mix_segments([seg], background_path, tmp_path / "final.wav", sample_rate=sr)
    mixed, _ = sf.read(str(out_path), always_2d=True)
    assert mixed.shape[1] == 2
    assert np.allclose(mixed[int(0.2 * sr), :], [0.1, -0.1], atol=1e-4)      # stereo image intact
    assert np.allclose(mixed[int(0.7 * sr), :], [0.3, 0.1], atol=1e-4)       # voice in both channels


def test_output_keeps_the_background_rate_and_the_voice_is_resampled_to_it(tmp_path):
    # Mixing used to happen at the TTS rate (24kHz), resampling the original
    # soundtrack down to it.
    bg_sr, voice_sr = 2000, 1000
    background = np.zeros(int(2.0 * bg_sr), dtype=np.float32)
    background_path = _write_wav(tmp_path / "background.wav", background, bg_sr)

    seg = Segment(start=0.5, end=1.0, speaker_id="A")
    seg.synthesized_audio = np.full(int(0.5 * voice_sr), 0.3, dtype=np.float32)

    out_path = mix_segments([seg], background_path, tmp_path / "final.wav", sample_rate=voice_sr)
    mixed, out_sr = sf.read(str(out_path))
    assert out_sr == bg_sr
    assert len(mixed) == len(background)
    # half a second of voice is still half a second at the new rate
    assert np.count_nonzero(np.abs(mixed) > 0.15) == pytest.approx(int(0.5 * bg_sr), abs=int(0.02 * bg_sr))


def test_clips_are_scaled_down_to_avoid_overflow(tmp_path):
    sr = 1000
    background = np.full(int(1.0 * sr), 0.9, dtype=np.float32)
    background_path = _write_wav(tmp_path / "background.wav", background, sr)

    seg = Segment(start=0.0, end=1.0, speaker_id="SPEAKER_00")
    seg.synthesized_audio = np.full(int(1.0 * sr), 0.9, dtype=np.float32)  # 0.9+0.9=1.8, would clip

    out_path = mix_segments([seg], background_path, tmp_path / "final.wav", sample_rate=sr)
    mixed, _ = sf.read(str(out_path))
    assert np.abs(mixed).max() <= 0.98 + 1e-6
