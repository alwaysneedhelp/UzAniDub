from dubtool.backends.transcribe_whisper import _speaker_at
from dubtool.types import Segment


def test_assigns_the_speaker_whose_turn_contains_the_midpoint():
    diarized = [
        Segment(start=0.0, end=5.0, speaker_id="SPEAKER_00"),
        Segment(start=5.0, end=10.0, speaker_id="SPEAKER_01"),
        Segment(start=10.0, end=15.0, speaker_id="SPEAKER_00"),
    ]

    assert _speaker_at(diarized, 2.5) == "SPEAKER_00"
    assert _speaker_at(diarized, 7.5) == "SPEAKER_01"
    assert _speaker_at(diarized, 12.0) == "SPEAKER_00"


def test_does_not_collapse_everything_onto_the_first_speaker():
    # the actual regression: an earlier version always returned
    # diarized[0].speaker_id regardless of `t`, which silently broke
    # multi-speaker diarization (every line ends up cloned in one voice).
    diarized = [
        Segment(start=0.0, end=5.0, speaker_id="SPEAKER_00"),
        Segment(start=5.0, end=10.0, speaker_id="SPEAKER_01"),
    ]

    assert _speaker_at(diarized, 7.5) != diarized[0].speaker_id


def test_falls_back_to_nearest_turn_in_a_gap():
    diarized = [
        Segment(start=0.0, end=5.0, speaker_id="SPEAKER_00"),
        Segment(start=8.0, end=12.0, speaker_id="SPEAKER_01"),
    ]

    # t=6.0 falls in the gap between turns, closer to SPEAKER_00's end (5.0)
    # than SPEAKER_01's start (8.0)
    assert _speaker_at(diarized, 6.0) == "SPEAKER_00"
    assert _speaker_at(diarized, 6.6) == "SPEAKER_01"


def test_empty_diarization_falls_back_to_default_speaker():
    assert _speaker_at([], 3.0) == "SPEAKER_00"
