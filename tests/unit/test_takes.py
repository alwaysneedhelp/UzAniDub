"""A synthesized take with no speech in it must never reach the mix.

Measured at 1 in 40 takes on real dialogue: CosyVoice2 reported success,
and the line would have played as noise -- heard as a skipped line.
"""
from pathlib import Path

import numpy as np

from dubtool.pipeline import _synthesize_take
from dubtool.stages.takes import is_usable, voiced_fraction
from dubtool.types import Segment

SR = 16000


def _voiced(seconds=1.5, freq=160.0):
    t = np.arange(int(seconds * SR)) / SR
    # a harmonic tone with a slow vibrato: periodic, as voiced speech is
    f = freq * (1 + 0.02 * np.sin(2 * np.pi * 3 * t))
    phase = 2 * np.pi * np.cumsum(f) / SR
    return (0.3 * (np.sin(phase) + 0.5 * np.sin(2 * phase))).astype(np.float32)


def _noise(seconds=1.5):
    return (0.2 * np.random.default_rng(0).standard_normal(int(seconds * SR))).astype(np.float32)


def test_voiced_audio_is_usable():
    usable, voiced = is_usable(_voiced(), SR, "happy")
    assert usable and voiced > 0.5


def test_unvoiced_noise_is_rejected():
    usable, voiced = is_usable(_noise(), SR, "happy")
    assert not usable and voiced < 0.15


def test_silence_is_rejected_even_for_a_whisper():
    assert is_usable(np.zeros(SR, dtype=np.float32), SR, "whispers")[0] is False


def test_a_whisper_is_not_held_to_the_voicing_test():
    # Whispered speech is unvoiced by design; a voicing test would reject
    # every legitimate whisper and re-roll it into a different style.
    assert is_usable(_noise(), SR, "whispers")[0] is True


def test_voiced_fraction_of_empty_audio_is_zero():
    assert voiced_fraction(np.zeros(0, dtype=np.float32), SR) == 0.0


class _ScriptedTTS:
    """Returns the queued takes in order and records what was asked for."""
    sample_rate = SR

    def __init__(self, takes):
        self._takes = list(takes)
        self.calls = []

    def synthesize(self, text, reference_audio, reference_text="", emotion=None, speed=1.0):
        self.calls.append(emotion)
        return self._takes.pop(0)


def _seg():
    return Segment(start=1.0, end=2.0, speaker_id="A", translated_text="Ha, bu to'g'ri")


def test_a_broken_take_is_re_rolled():
    tts = _ScriptedTTS([_noise(), _voiced()])
    out = _synthesize_take(tts, SR, _seg(), Path("ref.wav"), "", "happy")
    assert len(tts.calls) == 2
    assert is_usable(out, SR, "happy")[0]


def test_the_last_attempt_falls_back_to_calm():
    tts = _ScriptedTTS([_noise(), _noise(), _voiced()])
    _synthesize_take(tts, SR, _seg(), Path("ref.wav"), "", "happy")
    assert tts.calls == ["happy", "happy", "calm"]


def test_a_clean_first_take_costs_nothing_extra():
    tts = _ScriptedTTS([_voiced()])
    _synthesize_take(tts, SR, _seg(), Path("ref.wav"), "", "angry")
    assert tts.calls == ["angry"]


def test_when_every_take_fails_the_most_speech_like_one_is_kept():
    silent = np.zeros(int(1.5 * SR), dtype=np.float32)
    tts = _ScriptedTTS([silent, _noise(), silent])
    out = _synthesize_take(tts, SR, _seg(), Path("ref.wav"), "", "happy")
    assert out is not None and float(np.abs(out).max()) > 0   # the noise take, not silence
