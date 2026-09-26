from pathlib import Path

import numpy as np
import soundfile as sf

from dubtool.stages.emotion import classify_segments
from dubtool.types import Segment


def _tone(freq: float, amplitude: float, duration: float, sr: int) -> np.ndarray:
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _build_vocals(tones: list[tuple[float, float, float]], sr: int, path: Path) -> list[Segment]:
    """tones: list of (freq, amplitude, duration). Returns matching Segments
    (all one speaker) and writes the concatenated audio to `path`."""
    segments = []
    chunks = []
    t = 0.0
    for freq, amplitude, duration in tones:
        chunks.append(_tone(freq, amplitude, duration, sr))
        segments.append(Segment(start=t, end=t + duration, speaker_id="SPEAKER_00"))
        t += duration
    sf.write(str(path), np.concatenate(chunks), sr)
    return segments


def test_loud_high_pitched_segment_is_not_classified_as_the_calm_baseline(tmp_path):
    sr = 16000
    # baseline: three normal segments around 150Hz, moderate amplitude
    # outlier: one much louder, much higher-pitched segment (excited/surprised-like)
    tones = [
        (150, 0.2, 2.0),
        (150, 0.2, 2.0),
        (150, 0.2, 2.0),
        (400, 0.6, 2.0),
    ]
    vocals_path = tmp_path / "vocals.wav"
    segments = _build_vocals(tones, sr, vocals_path)

    result = classify_segments(segments, vocals_path)

    assert 3 in result
    assert result[3] in ("happy", "surprised")
    # baseline-ish segments get the neutral preset, not a strong emotion --
    # but they're still classified (routed through instruct2), not left
    # out entirely for zero-shot to handle (see stages/emotion.py).
    assert result[0] == "calm"


def test_quiet_segment_is_classified_as_whispers(tmp_path):
    sr = 16000
    tones = [
        (150, 0.3, 2.0),
        (150, 0.3, 2.0),
        (150, 0.05, 2.0),  # much quieter than baseline
    ]
    vocals_path = tmp_path / "vocals.wav"
    segments = _build_vocals(tones, sr, vocals_path)

    result = classify_segments(segments, vocals_path)

    assert result.get(2) == "whispers"


def test_uniform_delivery_is_classified_as_calm(tmp_path):
    sr = 16000
    tones = [(150, 0.3, 2.0)] * 4
    vocals_path = tmp_path / "vocals.wav"
    segments = _build_vocals(tones, sr, vocals_path)

    result = classify_segments(segments, vocals_path)

    assert result == {0: "calm", 1: "calm", 2: "calm", 3: "calm"}


def test_silent_segment_is_left_unclassified(tmp_path):
    # no usable pitch signal at all -- the one remaining case that should
    # fall back to zero-shot rather than get an explicit style.
    sr = 16000
    tones = [(150, 0.3, 2.0), (150, 0.3, 2.0), (0, 0.0, 2.0)]
    vocals_path = tmp_path / "vocals.wav"
    segments = _build_vocals(tones, sr, vocals_path)

    result = classify_segments(segments, vocals_path)

    assert 2 not in result


def test_keeps_speakers_independent(tmp_path):
    sr = 16000
    # speaker A: consistently loud; speaker B: consistently quiet.
    # Neither should be flagged as strongly emotional just for being
    # naturally louder/quieter than the other -- baselines are per-speaker,
    # so both should land on the neutral "calm" preset, not e.g. "angry"/
    # "whispers" from comparing across speakers.
    segments = [
        Segment(start=0.0, end=2.0, speaker_id="A"),
        Segment(start=2.0, end=4.0, speaker_id="B"),
        Segment(start=4.0, end=6.0, speaker_id="A"),
        Segment(start=6.0, end=8.0, speaker_id="B"),
    ]
    t = np.linspace(0, 8.0, 8 * sr, endpoint=False)
    data = np.zeros_like(t)
    data[0 * sr : 2 * sr] = 0.5 * np.sin(2 * np.pi * 150 * t[0 * sr : 2 * sr])   # A, loud
    data[2 * sr : 4 * sr] = 0.1 * np.sin(2 * np.pi * 150 * t[2 * sr : 4 * sr])   # B, quiet
    data[4 * sr : 6 * sr] = 0.5 * np.sin(2 * np.pi * 150 * t[4 * sr : 6 * sr])   # A, loud (same as baseline)
    data[6 * sr : 8 * sr] = 0.1 * np.sin(2 * np.pi * 150 * t[6 * sr : 8 * sr])   # B, quiet (same as baseline)

    vocals_path = tmp_path / "vocals.wav"
    sf.write(str(vocals_path), data.astype(np.float32), sr)

    result = classify_segments(segments, vocals_path)

    assert result == {0: "calm", 1: "calm", 2: "calm", 3: "calm"}
