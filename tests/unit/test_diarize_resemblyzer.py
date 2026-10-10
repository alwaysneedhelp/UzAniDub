"""The diarizer must account for every bit of speech it is given.

Demucs removes the original voices before mixing, so any speech the
diarizer fails to emit a turn for is not merely mis-labelled -- it reaches
the finished dub as background music over a character whose mouth is
moving. On a real clip an earlier version of this backend left 36% of all
dialogue uncovered, which is what "it skips lines" turned out to mean.

Coverage therefore comes from the VAD speech mask and identity from the
clustered embedding windows. These tests pin that split: a speech region
with no surviving window of its own still gets dubbed.
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from dubtool.backends.diarize_resemblyzer import _SAMPLE_RATE, ResemblyzerDiarizer


def _speechlike(path, segments, duration=10.0, sr=_SAMPLE_RATE):
    """Noise bursts the VAD will accept, separated by true silence."""
    rng = np.random.default_rng(0)
    wav = np.zeros(int(duration * sr), dtype=np.float32)
    for start, end, pitch in segments:
        n = int((end - start) * sr)
        t = np.arange(n) / sr
        tone = 0.3 * np.sin(2 * np.pi * pitch * t)
        tone += 0.05 * rng.standard_normal(n).astype(np.float32)
        wav[int(start * sr):int(start * sr) + n] = tone
    sf.write(str(path), wav, sr)
    return path


def test_every_detected_speech_sample_lands_in_some_turn(tmp_path):
    path = _speechlike(tmp_path / "v.wav", [(0.5, 3.0, 140), (4.0, 6.5, 240), (7.0, 9.5, 140)])
    diarizer = ResemblyzerDiarizer()
    wav = diarizer._load_16k_mono(path)
    mask = diarizer._speech_mask(wav)

    segments = diarizer.diarize(path, num_speakers=2)

    covered = np.zeros(len(wav), dtype=bool)
    for seg in segments:
        covered[int(seg.start * _SAMPLE_RATE):int(seg.end * _SAMPLE_RATE)] = True
    uncovered = (mask & ~covered).sum() / _SAMPLE_RATE
    assert uncovered < 0.1, f"{uncovered:.2f}s of speech would never be dubbed"


def test_turns_are_disjoint_and_ordered(tmp_path):
    # Downstream treats turns as spans to transcribe and mix independently;
    # overlapping ones double-count the same audio.
    path = _speechlike(tmp_path / "v.wav", [(0.5, 3.0, 140), (3.8, 6.0, 240)])
    segments = ResemblyzerDiarizer().diarize(path, num_speakers=2)

    assert segments == sorted(segments, key=lambda s: s.start)
    for earlier, later in zip(segments, segments[1:]):
        assert later.start >= earlier.end - 1e-6


def test_no_turn_is_too_short_to_say_anything_in(tmp_path):
    path = _speechlike(tmp_path / "v.wav", [(0.5, 3.0, 140), (4.0, 6.5, 240)])
    segments = ResemblyzerDiarizer().diarize(path, num_speakers=2)

    assert segments
    assert all((seg.end - seg.start) >= 0.4 - 1e-6 for seg in segments)


def test_silence_yields_a_single_fallback_turn_not_an_empty_dub(tmp_path):
    sf.write(str(tmp_path / "s.wav"), np.zeros(int(3.0 * _SAMPLE_RATE), dtype=np.float32), _SAMPLE_RATE)
    segments = ResemblyzerDiarizer().diarize(tmp_path / "s.wav")

    assert len(segments) == 1
    assert segments[0].speaker_id == "SPEAKER_00"


def test_requested_speaker_count_is_respected(tmp_path):
    path = _speechlike(tmp_path / "v.wav", [(0.5, 3.0, 140), (4.0, 6.5, 240), (7.0, 9.5, 140)])
    segments = ResemblyzerDiarizer().diarize(path, num_speakers=2)

    assert len({seg.speaker_id for seg in segments}) <= 2
