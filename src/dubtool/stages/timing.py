"""Decides when each dubbed line starts and how fast it is spoken, so that
every line is heard to its last word.

The earlier approach fitted each line into its *own* original slot: speed
it up a little, phase-vocoder stretch it a little more, and let mix.py cut
off whatever still ran over at the next line's start. Uzbek routinely runs
longer than the English it replaces, so on a real clip 13 of 16 lines were
cut mid-word, and the stretching put most lines through a phase vocoder
for a 5% gain -- the metallic, phasey quality that reads as "robotic".

So the priorities are now explicit, in order:
  1. Every line plays in full. Nothing is truncated, ever.
  2. A line may use the pause after it, not just its own slot: the room it
     really has is until the *next* line is due.
  3. Only then speed it up, through the TTS's own speed control (which
     regenerates the speech) rather than stretching a finished waveform,
     and only as much as the room requires, up to a cap.
  4. If it still doesn't fit, the next line waits for it.
  5. Once a line is running late, the speed cap rises, so the lag is worked
     off rather than left to pile up. Waiting for pauses alone doesn't do
     it: MADLAD's Uzbek ran 45s of natural speech for a 37.5s clip, dense
     dialogue has few pauses, and at a flat 1.25x cap the lag reached 3.4s
     -- lines landing in the wrong shot. Simulated on those real line
     lengths, 1.25x on time rising to 1.40x past 0.3s late held the worst
     lag to 1.1s at a mean speed of 1.26x.

Both the synthesis planner in pipeline.py and the mixer use place() below,
so the placement a speed was chosen for is the placement that is heard.
"""
from __future__ import annotations

# A short line ending this far ahead of the original is slowed slightly so
# the character isn't left mouthing silence; above it, natural pace wins.
_FILL_THRESHOLD = 0.85


def place(original_start: float, previous_end: float, min_gap: float) -> float:
    """When a line actually begins: at its original time, unless the line
    before it is still being spoken -- then just after it."""
    return max(original_start, previous_end + min_gap)


def room(start: float, next_due: float) -> float:
    """How long a line can run before the next one is due to start."""
    return next_due - start


def choose_speed(
    natural_duration: float,
    available: float,
    original_duration: float,
    min_speed: float,
    max_speed: float,
) -> float:
    """Speed to re-synthesize a line at, given how long it naturally takes.

    >1 speeds speech up, <1 slows it down, 1.0 means keep the natural take.
    """
    if natural_duration <= 0:
        return 1.0
    if available <= 0:
        # Already running late (the previous line pushed this one past when
        # the next was due): catch up as fast as is still allowed.
        return max_speed
    if natural_duration > available:
        return min(max_speed, natural_duration / available)
    if original_duration > 0 and natural_duration < original_duration * _FILL_THRESHOLD:
        return max(min_speed, natural_duration / original_duration)
    return 1.0


def speed_cap(lag: float, on_time_cap: float, catchup_cap: float, catchup_after: float) -> float:
    """The fastest a line may be spoken. Gentle while lines are on time;
    higher once this line is starting more than `catchup_after` seconds
    late, so the delay shrinks instead of accumulating line after line."""
    return catchup_cap if lag > catchup_after else on_time_cap
