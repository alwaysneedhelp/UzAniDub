import pytest
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


# --- following the actor ---------------------------------------------------
import soundfile as _sf
from dubtool.stages.loudness import active_rms, actor_targets, match_level
from dubtool.types import Segment as _Segment

_SR = 16000


def _sine(seconds, amp, freq=200.0):
    t = np.arange(int(seconds * _SR)) / _SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_active_level_ignores_the_pauses_inside_a_line():
    speech = _sine(1.0, 0.3)
    with_pause = np.concatenate([speech, np.zeros(_SR, dtype=np.float32)])
    assert active_rms(with_pause, _SR) == pytest.approx(active_rms(speech, _SR), rel=0.02)


def test_a_line_is_matched_to_the_level_it_is_given():
    out = match_level(_sine(1.0, 0.05), _SR, target_active_rms=0.1)
    assert active_rms(out, _SR) == pytest.approx(0.1, rel=0.02)


def test_a_loud_line_is_limited_on_its_own_rather_than_clipping():
    out = match_level(_sine(1.0, 0.5), _SR, target_active_rms=0.9)
    assert float(np.abs(out).max()) <= 0.95 + 1e-6


def test_a_shout_with_sharp_peaks_still_reaches_its_level():
    # Capping gain at ceiling/peak let one transient hold a whole shouted
    # line down: +1.1dB over an ordinary line instead of the actor's +5.5dB.
    shout = _sine(1.0, 0.2)
    shout[::400] = 0.9                       # sparse sharp transients
    out = match_level(shout, _SR, target_active_rms=0.3)
    assert float(np.abs(out).max()) <= 0.95 + 1e-6
    assert 20 * np.log10(active_rms(out, _SR) / 0.3) > -1.0   # within 1dB of target


def test_the_actors_shout_stays_louder_than_their_ordinary_lines(tmp_path):
    # The fixed preset levels squeezed a 9.8dB range into 2.3dB; a shout sat
    # 5.7dB above a normal line in the original and 1.5dB in the dub.
    audio = np.concatenate([_sine(1.0, 0.1), _sine(1.0, 0.1), _sine(1.0, 0.1), _sine(1.0, 0.2)])
    path = tmp_path / "vocals.wav"
    _sf.write(str(path), audio, _SR)
    segs = [_Segment(start=float(i), end=float(i + 1), speaker_id="A") for i in range(4)]

    targets = actor_targets(segs, path)

    assert 20 * np.log10(targets[3] / targets[0]) == pytest.approx(6.0, abs=0.3)


def test_a_line_far_off_the_scene_level_is_clamped(tmp_path):
    audio = np.concatenate([_sine(1.0, 0.1)] * 3 + [_sine(1.0, 0.9)])
    path = tmp_path / "vocals.wav"
    _sf.write(str(path), audio, _SR)
    segs = [_Segment(start=float(i), end=float(i + 1), speaker_id="A") for i in range(4)]

    targets = actor_targets(segs, path)

    assert targets[3] / targets[0] == pytest.approx(3.0, rel=0.02)


def test_a_silent_line_has_no_target(tmp_path):
    audio = np.concatenate([_sine(1.0, 0.1), np.zeros(_SR, dtype=np.float32)])
    path = tmp_path / "vocals.wav"
    _sf.write(str(path), audio, _SR)
    segs = [_Segment(start=0.0, end=1.0, speaker_id="A"), _Segment(start=1.0, end=2.0, speaker_id="A")]

    assert actor_targets(segs, path)[1] is None


# --- vocal effort -------------------------------------------------------------
from dubtool.stages.loudness import effort_for, vocal_effort


def test_ordinary_and_mildly_raised_lines_get_no_effort():
    assert effort_for(0.0) == 0.0
    assert effort_for(2.5) == 0.0


def test_a_scream_gets_most_of_the_effort():
    # The screamed line on the test clip sat +5.5dB over the scene.
    assert effort_for(5.5) == pytest.approx(5.1, abs=0.1)
    assert effort_for(12.0) == 6.0


def _band_share(y, sr, lo=2000, hi=4000):
    spec = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / sr)
    return 10 * np.log10(spec[(f >= lo) & (f < hi)].sum() / spec.sum())


def test_effort_lifts_the_presence_band_where_a_shout_has_its_edge():
    rng = np.random.default_rng(0)
    noise = (0.1 * rng.standard_normal(24000)).astype(np.float32)
    assert _band_share(vocal_effort(noise, 24000, 6.0), 24000) > _band_share(noise, 24000) + 2.0


def test_no_effort_leaves_the_line_untouched():
    x = _sine(0.5, 0.2)
    assert vocal_effort(x, _SR, 0.0) is x
