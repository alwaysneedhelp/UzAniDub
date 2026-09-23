from pathlib import Path

import numpy as np
import soundfile as sf

from dubtool.stages.reference import extract_reference_clips, extract_segment_reference
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
    picked, picked_sr = sf.read(str(clips["SPEAKER_00"].path), dtype="float32")
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

    picked, _ = sf.read(str(clips["SPEAKER_00"].path), dtype="float32")
    # picked should match the *raw* signal closely, not the noisy "separated" one
    start = 0  # loudness is uniform here, so the window should start near 0
    raw_window = raw[start : start + len(picked)]
    assert np.allclose(picked, raw_window, atol=1e-4)


def test_prefers_a_dynamic_window_over_an_equally_loud_flat_one(tmp_path):
    # the real regression this guards against: a pitch-variance check on
    # real output showed the extracted reference clip was much flatter than
    # the character's actual delivery — the old scoring (rms * non-silence
    # only) can't distinguish an expressive stretch from a uniformly loud
    # monotone one, so it doesn't reliably avoid picking the flat one.
    sr = 16000
    duration = 20.0
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)

    # 0-8s: flat, constant-amplitude tone (same RMS throughout)
    flat = 0.3 * np.sin(2 * np.pi * 220 * t[: 8 * sr])
    # 12-20s: same average amplitude, but bursty — alternating loud/soft
    # half-second bursts, closer to real expressive speech dynamics
    dynamic = np.zeros(8 * sr, dtype=np.float32)
    burst_len = sr // 2
    for i in range(0, 8 * sr, burst_len):
        amplitude = 0.5 if (i // burst_len) % 2 == 0 else 0.1
        dynamic[i : i + burst_len] = amplitude * np.sin(2 * np.pi * 220 * t[i : i + burst_len])

    data = np.zeros(int(duration * sr), dtype=np.float32)
    data[: 8 * sr] = flat
    data[12 * sr : 20 * sr] = dynamic

    vocals_path = _write_wav(tmp_path / "vocals.wav", data, sr)
    segments = [Segment(start=0.0, end=duration, speaker_id="SPEAKER_00")]

    clips = extract_reference_clips(vocals_path, segments, tmp_path, target_duration=4.0, min_duration=2.0, hop=0.5)

    picked, _ = sf.read(str(clips["SPEAKER_00"].path), dtype="float32")
    # the picked window should fall in the dynamic region (12-20s), not the flat one (0-8s)
    assert not np.allclose(picked, flat[: len(picked)], atol=1e-3), (
        "picked the flat window over the equally-loud dynamic one"
    )


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

    picked, _ = sf.read(str(clips["SPEAKER_00"].path), dtype="float32")
    # should stay with the clean separated vocals, not the raw+background mix
    assert not np.allclose(picked, raw[: len(picked)], atol=1e-3)


def test_segment_reference_uses_the_segments_own_audio_and_text(tmp_path):
    sr = 16000
    data = 0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 10.0, 10 * sr, endpoint=False)).astype(np.float32)
    vocals_path = _write_wav(tmp_path / "vocals.wav", data, sr)
    seg = Segment(start=2.0, end=6.0, speaker_id="SPEAKER_00", text="an angry shout")

    clip = extract_segment_reference(seg, vocals_path, tmp_path / "ref.wav")

    assert clip is not None
    assert clip.text == "an angry shout"
    picked, picked_sr = sf.read(str(clip.path), dtype="float32")
    assert picked_sr == sr
    expected = data[int(2.0 * sr) : int(6.0 * sr)]
    assert np.allclose(picked, expected, atol=1e-4)


def test_segment_reference_returns_none_when_too_short(tmp_path):
    sr = 16000
    data = np.zeros(int(3.0 * sr), dtype=np.float32)
    vocals_path = _write_wav(tmp_path / "vocals.wav", data, sr)
    seg = Segment(start=0.0, end=0.5, speaker_id="SPEAKER_00", text="hi")  # below min_duration

    assert extract_segment_reference(seg, vocals_path, tmp_path / "ref.wav", min_duration=1.5) is None


def test_segment_reference_returns_none_when_clipped(tmp_path):
    sr = 16000
    data = np.ones(int(3.0 * sr), dtype=np.float32)  # peak at 1.0 -> clipped
    vocals_path = _write_wav(tmp_path / "vocals.wav", data, sr)
    seg = Segment(start=0.0, end=3.0, speaker_id="SPEAKER_00", text="clipped audio")

    assert extract_segment_reference(seg, vocals_path, tmp_path / "ref.wav") is None


def test_segment_reference_caps_at_max_duration(tmp_path):
    sr = 16000
    data = 0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 20.0, 20 * sr, endpoint=False)).astype(np.float32)
    vocals_path = _write_wav(tmp_path / "vocals.wav", data, sr)
    seg = Segment(start=0.0, end=20.0, speaker_id="SPEAKER_00", text="a very long line")

    clip = extract_segment_reference(seg, vocals_path, tmp_path / "ref.wav", max_duration=5.0)

    assert clip is not None
    assert abs((clip.end - clip.start) - 5.0) < 1e-6
