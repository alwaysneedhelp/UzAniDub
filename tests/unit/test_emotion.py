from pathlib import Path

import numpy as np
import soundfile as sf

from dubtool.stages.emotion import _classify, classify_segments
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


def test_a_shout_is_angry_whatever_its_punctuation():
    assert _classify(energy_vs_scene=1.9, pitch_ratio=1.0, text="You're so right!") == "angry"


def test_an_exclamation_gets_energetic_delivery_not_calm():
    # "I could never do that till now!" was classified calm before.
    assert _classify(energy_vs_scene=1.3, pitch_ratio=None, text="I could never do that till now!") == "happy"


def test_a_quiet_reaction_question_is_surprise_not_a_whisper():
    # "Are you for real?" was whispered for being quiet.
    assert _classify(energy_vs_scene=0.63, pitch_ratio=None, text="Are you for real?") == "surprised"


def test_a_long_informational_question_stays_calm():
    text = "Where exactly did you say the two of them were planning to meet tonight?"
    assert _classify(energy_vs_scene=0.8, pitch_ratio=1.0, text=text) == "calm"


def test_a_quiet_plain_line_is_whispered():
    assert _classify(energy_vs_scene=0.3, pitch_ratio=1.0, text="Stay down.") == "whispers"


def test_full_width_punctuation_counts():
    assert _classify(energy_vs_scene=1.0, pitch_ratio=None, text="待て！") == "happy"


def test_low_pitch_alone_does_not_make_a_line_sad():
    # "Reverse Cursed Technique." came out sorrowful: ordinary loudness, a
    # lowish pitch against the speaker's median. Sorrow needs it quiet too.
    assert _classify(energy_vs_scene=0.69, pitch_ratio=0.62, text="Reverse Cursed Technique.") == "calm"


def test_quiet_and_low_is_sad():
    assert _classify(energy_vs_scene=0.5, pitch_ratio=0.8, text="I couldn't save them.") == "sad"


def test_pitch_variability_no_longer_selects_nervous_or_tired():
    # It fired on an animated explanation as easily as on nerves.
    assert _classify(energy_vs_scene=0.98, pitch_ratio=1.09, text="So you multiply it against itself.") == "calm"


def test_ordinary_delivery_stays_calm():
    assert _classify(energy_vs_scene=1.0, pitch_ratio=1.0, text="Our fight is just getting started.") == "calm"


def test_loudness_is_judged_against_the_scene_not_each_speaker(tmp_path):
    # Per-speaker baselines made a character who shouts throughout "calm":
    # their median line *is* a shout. Dialogue in a mixed dub is levelled
    # across characters, so a character far louder than everyone else is
    # shouting, and one far quieter is murmuring.
    sr = 16000
    segments = [
        Segment(start=0.0, end=2.0, speaker_id="A", text="Get out of my way"),
        Segment(start=2.0, end=4.0, speaker_id="B", text="okay then"),
        Segment(start=4.0, end=6.0, speaker_id="C", text="we should go"),
        Segment(start=6.0, end=8.0, speaker_id="A", text="Move it now"),
        Segment(start=8.0, end=10.0, speaker_id="C", text="right behind you"),
    ]
    t = np.linspace(0, 10.0, 10 * sr, endpoint=False)
    data = np.zeros_like(t)
    for (a, b), amp in zip([(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)], [0.6, 0.04, 0.2, 0.6, 0.2]):
        data[a * sr:b * sr] = amp * np.sin(2 * np.pi * 150 * t[a * sr:b * sr])
    vocals_path = tmp_path / "vocals.wav"
    sf.write(str(vocals_path), data.astype(np.float32), sr)

    result = classify_segments(segments, vocals_path)

    assert result[0] == "angry" and result[3] == "angry"   # A shouts both lines
    assert result[1] == "whispers"                           # B is far below the scene
    assert result[2] == "calm" and result[4] == "calm"


def test_a_line_pitch_tracking_missed_is_still_classified_from_loudness_and_punctuation(tmp_path):
    # "Exactly!" was left unclassified because pyin found no pitch in it.
    sr = 16000
    rng = np.random.default_rng(0)
    segments = [
        Segment(start=0.0, end=2.0, speaker_id="A", text="fine"),
        Segment(start=2.0, end=4.0, speaker_id="A", text="sure"),
        Segment(start=4.0, end=5.0, speaker_id="A", text="Exactly!"),
    ]
    data = np.zeros(5 * sr, dtype=np.float32)
    t = np.arange(4 * sr) / sr
    data[:4 * sr] = 0.2 * np.sin(2 * np.pi * 150 * t)
    data[4 * sr:] = (0.2 * rng.standard_normal(sr)).astype(np.float32)   # unpitched
    vocals_path = tmp_path / "vocals.wav"
    sf.write(str(vocals_path), data, sr)

    assert classify_segments(segments, vocals_path).get(2) == "happy"


def test_a_shout_far_above_the_speakers_usual_pitch_is_a_shriek_not_anger():
    assert _classify(energy_vs_scene=2.5, pitch_ratio=1.8, text="") == "surprised"
    assert _classify(energy_vs_scene=2.5, pitch_ratio=1.2, text="") == "angry"
