"""Lays the dubbed lines over the separated background (music + effects)
to produce the final soundtrack."""
from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from dubtool.stages import timing
from dubtool.types import Segment

log = logging.getLogger(__name__)


def mix_segments(
    segments: list[Segment],
    background_path: Path,
    out_path: Path,
    sample_rate: int,
    min_gap: float = 0.08,
) -> Path:
    """Places every line, in full, over the background track.

    `sample_rate` is the rate of the synthesized lines. The output keeps the
    *background's* rate and channel count -- the background is the original
    music and effects, and it used to be downmixed to mono and resampled to
    the TTS's 24kHz before the voice went on top, which dulled the whole
    soundtrack next to the original. The voice is resampled up to meet it
    instead, and sits in the centre of every channel.

    Nothing is ever truncated. A line starts at `placed_start` (set by the
    pipeline's timing planner) or its original `start`, and if the line
    before it is still being spoken it waits -- the same rule the planner
    used (timing.place), so the start a speed was chosen for is the start
    that's heard. This replaces cutting a line off at the next line's start,
    which is what made dubbed dialogue sound like it was skipping words.
    """
    background, bg_sr = sf.read(str(background_path), dtype="float32", always_2d=True)

    placements: list[tuple[int, np.ndarray]] = []
    previous_end = float("-inf")
    for seg in sorted(segments, key=lambda s: s.start):
        if seg.synthesized_audio is None or len(seg.synthesized_audio) == 0:
            continue
        voice = seg.synthesized_audio.astype(np.float32)
        if sample_rate != bg_sr:
            voice = librosa.resample(voice, orig_sr=sample_rate, target_sr=bg_sr).astype(np.float32)

        wanted = seg.placed_start if seg.placed_start is not None else seg.start
        start = timing.place(wanted, previous_end, min_gap)
        if start - wanted > 0.01:
            log.info(
                "line at %.2fs moved to %.2fs so the line before it can finish",
                wanted, start,
            )
        placements.append((int(round(start * bg_sr)), voice))
        previous_end = start + len(voice) / bg_sr

    total = max(len(background), max((s + len(v) for s, v in placements), default=0))
    vocal = np.zeros(total, dtype=np.float32)
    for start_sample, voice in placements:
        vocal[start_sample:start_sample + len(voice)] += voice
    if len(background) < total:
        background = np.pad(background, ((0, total - len(background)), (0, 0)))

    mixed = background + vocal[:, None]
    peak = float(np.abs(mixed).max()) if mixed.size else 0.0
    if peak > 0.98:
        log.warning("mixed output peak %.3f exceeds 0.98; scaling down to avoid clipping", peak)
        mixed = mixed * (0.98 / peak)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), mixed[:, 0] if mixed.shape[1] == 1 else mixed, bg_sr)
    return out_path
