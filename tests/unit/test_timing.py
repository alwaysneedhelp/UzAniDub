"""The rules that decide when a dubbed line starts and how fast it is spoken.

Priorities, in order: every line is heard in full; a line may use the
pause after it; only then is it sped up (gently, through the TTS); and if
it still runs long, the next line waits rather than cutting it off.
"""
import pytest

from dubtool.stages.timing import choose_speed, place, room


def test_a_line_starts_on_time_when_the_previous_one_has_finished():
    assert place(original_start=5.0, previous_end=3.0, min_gap=0.08) == 5.0


def test_a_line_waits_for_the_previous_one_rather_than_talking_over_it():
    assert place(original_start=5.0, previous_end=5.4, min_gap=0.08) == pytest.approx(5.48)


def test_a_line_never_starts_earlier_than_its_original_time():
    # This is what lets accumulated delay recover at the next pause.
    assert place(original_start=5.0, previous_end=float("-inf"), min_gap=0.08) == 5.0


def test_room_includes_the_pause_after_the_line():
    # Slot is 1.0s, but the next line isn't due until 1.6s later: that pause
    # is usable, and ignoring it is what forced lines to be cut.
    assert room(start=2.0, next_due=3.6) == pytest.approx(1.6)


def test_a_line_that_fits_its_room_is_left_at_natural_pace():
    assert choose_speed(1.4, available=1.6, original_duration=1.5, min_speed=0.9, max_speed=1.25) == 1.0


def test_a_line_slightly_too_long_is_sped_up_only_as_much_as_needed():
    speed = choose_speed(2.0, available=1.8, original_duration=1.5, min_speed=0.9, max_speed=1.25)
    assert speed == pytest.approx(2.0 / 1.8)


def test_speed_up_is_capped_and_the_rest_is_left_to_the_next_line_waiting():
    assert choose_speed(4.0, available=1.0, original_duration=1.0, min_speed=0.9, max_speed=1.25) == 1.25


def test_a_line_already_running_late_catches_up_as_fast_as_allowed():
    assert choose_speed(1.0, available=-0.3, original_duration=1.0, min_speed=0.9, max_speed=1.25) == 1.25


def test_a_line_much_shorter_than_the_original_is_slowed_gently_to_fill():
    # Leaves the character mouthing silence otherwise.
    speed = choose_speed(1.0, available=3.0, original_duration=2.0, min_speed=0.9, max_speed=1.25)
    assert speed == pytest.approx(0.9)


def test_a_line_only_slightly_shorter_than_the_original_is_left_alone():
    assert choose_speed(1.8, available=3.0, original_duration=2.0, min_speed=0.9, max_speed=1.25) == 1.0


def test_an_empty_take_is_not_resynthesized():
    assert choose_speed(0.0, available=1.0, original_duration=1.0, min_speed=0.9, max_speed=1.25) == 1.0


def test_the_cap_stays_gentle_while_lines_are_on_time():
    from dubtool.stages.timing import speed_cap
    assert speed_cap(lag=0.1, on_time_cap=1.25, catchup_cap=1.4, catchup_after=0.3) == 1.25


def test_the_cap_rises_once_a_line_is_running_late_so_lag_cannot_pile_up():
    # At a flat 1.25x on real dialogue the lag reached 3.4s: lines in the
    # wrong shot. Raising the cap only while late held it to about 1.1s.
    from dubtool.stages.timing import speed_cap
    assert speed_cap(lag=0.6, on_time_cap=1.25, catchup_cap=1.4, catchup_after=0.3) == 1.4
