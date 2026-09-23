from pathlib import Path

import numpy as np
import soundfile as sf

from dubtool.stages.reference import extract_reference_clips
from dubtool.types import Segment


def _write_wav(path: Path, data: np.ndarray, sr: int = 16000) -> Path:
    sf.write(str(path), data.astype(np.float32), sr)
    return path


def test_picks_the_loud_window_over_silence(tmp_path):
    sr = 16000
    duration = 12.0
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    data = np.zeros_like(t)
    # loud tone from 6s-10s, silence everywhere else
    loud_mask = (t >= 6.0) & (t < 10.0)
    data[loud_mask] = 0.5 * np.sin(2 * np.pi * 220 * t[loud_mask])

    vocals_path = _write_wav(tmp_path / "vocals.wav", data, sr)
    segments = [Segment(start=0.0, end=duration, speaker_id="SPEAKER_00")]

    clips = extract_reference_clips(vocals_path, segments, tmp_path, target_duration=3.0, min_duration=2.0)

    assert "SPEAKER_00" in clips
    picked, picked_sr = sf.read(str(clips["SPEAKER_00"]), dtype="float32")
    assert picked_sr == sr
    # the picked window should overlap the loud region, not be near-silent
    assert float(np.sqrt(np.mean(picked**2))) > 0.05


def test_returns_no_clip_when_speaker_has_too_little_speech(tmp_path):
    sr = 16000
    data = np.zeros(int(1.0 * sr), dtype=np.float32)  # 1s total, below min_duration
    vocals_path = _write_wav(tmp_path / "vocals.wav", data, sr)
    segments = [Segment(start=0.0, end=1.0, speaker_id="SPEAKER_00")]

    clips = extract_reference_clips(vocals_path, segments, tmp_path, target_duration=8.0, min_duration=4.0)

    assert clips == {}


def test_separates_clips_per_speaker(tmp_path):
    sr = 16000
    duration = 6.0
    data = 0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, duration, int(duration * sr), endpoint=False))
    vocals_path = _write_wav(tmp_path / "vocals.wav", data.astype(np.float32), sr)
    segments = [
        Segment(start=0.0, end=3.0, speaker_id="SPEAKER_00"),
        Segment(start=3.0, end=6.0, speaker_id="SPEAKER_01"),
    ]

    clips = extract_reference_clips(vocals_path, segments, tmp_path, target_duration=2.0, min_duration=1.0)

    assert set(clips) == {"SPEAKER_00", "SPEAKER_01"}
    assert clips["SPEAKER_00"] != clips["SPEAKER_01"]


def test_substitutes_raw_audio_when_background_energy_is_negligible(tmp_path):
    sr = 16000
    duration = 8.0
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    voice = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    # Simulate a Demucs artifact: the "separated" vocals differ slightly
    # from the true raw voice (a bit of added noise), while the background
    # track is essentially silent (nothing was actually there to remove).
    rng = np.random.default_rng(0)
    separated_vocals = voice + 0.05 * rng.standard_normal(len(voice)).astype(np.float32)
    background = np.zeros_like(voice)
    raw = voice  # the untouched original — what we want picked instead

    vocals_path = _write_wav(tmp_path / "vocals.wav", separated_vocals, sr)
    background_path = _write_wav(tmp_path / "background.wav", background, sr)
    raw_path = _write_wav(tmp_path / "full_audio.wav", raw, sr)
    segments = [Segment(start=0.0, end=duration, speaker_id="SPEAKER_00")]

    clips = extract_reference_clips(
        vocals_path, segments, tmp_path, target_duration=3.0, min_duration=2.0,
        raw_audio_path=raw_path, background_path=background_path,
    )

    picked, _ = sf.read(str(clips["SPEAKER_00"]), dtype="float32")
    # picked should match the *raw* signal closely, not the noisy "separated" one
    start = 0  # loudness is uniform here, so the window should start near 0
    raw_window = raw[start : start + len(picked)]
    assert np.allclose(picked, raw_window, atol=1e-4)


def test_keeps_separated_vocals_when_background_energy_is_significant(tmp_path):
    sr = 16000
    duration = 8.0
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    voice = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    separated_vocals = voice.copy()
    background = (0.2 * np.sin(2 * np.pi * 90 * t)).astype(np.float32)  # loud, real background
    raw = voice + background  # what the mic actually picked up

    vocals_path = _write_wav(tmp_path / "vocals.wav", separated_vocals, sr)
    background_path = _write_wav(tmp_path / "background.wav", background, sr)
    raw_path = _write_wav(tmp_path / "full_audio.wav", raw, sr)
    segments = [Segment(start=0.0, end=duration, speaker_id="SPEAKER_00")]

    clips = extract_reference_clips(
        vocals_path, segments, tmp_path, target_duration=3.0, min_duration=2.0,
        raw_audio_path=raw_path, background_path=background_path,
    )

    picked, _ = sf.read(str(clips["SPEAKER_00"]), dtype="float32")
    # should stay with the clean separated vocals, not the raw+background mix
    assert not np.allclose(picked, raw[: len(picked)], atol=1e-3)
