import numpy as np

from dubtool.stages.loudness import normalize


def _tone(rms_amplitude: float, duration: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    # for a sine wave, rms = amplitude / sqrt(2), so scale to hit a target rms exactly
    amplitude = rms_amplitude * np.sqrt(2)
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_quiet_segment_gets_boosted_to_target():
    audio = _tone(rms_amplitude=0.03)  # needs ~3.3x gain, comfortably under the 6x cap
    out = normalize(audio)
    assert abs(np.sqrt(np.mean(out**2)) - 0.1) < 0.005


def test_loud_segment_gets_reduced_to_target():
    audio = _tone(rms_amplitude=0.3)
    out = normalize(audio)
    assert abs(np.sqrt(np.mean(out**2)) - 0.1) < 0.005


def test_whispers_emotion_uses_a_lower_target_than_default():
    audio = _tone(rms_amplitude=0.2)
    normal = normalize(audio, emotion=None)
    whispered = normalize(audio, emotion="whispers")
    assert np.sqrt(np.mean(whispered**2)) < np.sqrt(np.mean(normal**2))


def test_angry_emotion_uses_a_higher_target_than_default():
    audio = _tone(rms_amplitude=0.02)
    normal = normalize(audio, emotion=None)
    angry = normalize(audio, emotion="angry")
    assert np.sqrt(np.mean(angry**2)) > np.sqrt(np.mean(normal**2))


def test_silence_is_left_untouched():
    audio = np.zeros(16000, dtype=np.float32)
    out = normalize(audio)
    assert np.allclose(out, audio)


def test_empty_array_is_left_untouched():
    audio = np.array([], dtype=np.float32)
    out = normalize(audio)
    assert len(out) == 0


def test_near_silent_segment_is_not_blown_up_past_the_gain_cap():
    # a tiny sliver of signal shouldn't get amplified into audible noise
    audio = _tone(rms_amplitude=0.001)
    out = normalize(audio)
    ratio = np.sqrt(np.mean(out**2)) / np.sqrt(np.mean(audio**2))
    assert ratio <= 6.0 + 1e-6
