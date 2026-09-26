"""Combines aligned per-segment synthesized speech with the separated
background track into one final mono audio file."""
from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from dubtool.types import Segment

log = logging.getLogger(__name__)


def mix_segments(
    segments: list[Segment],
    background_path: Path,
    out_path: Path,
    sample_rate: int,
    fade_ms: float = 15.0,
) -> Path:
    """Places each segment's aligned synthesized audio at its original
    timestamp on a silent timeline, then sums with the separated background
    track (resampled/downmixed to `sample_rate`, the TTS output rate, which
    is the pipeline's working rate for the final mix).

    Step 2 downmixes background to mono for simplicity; preserving stereo
    background is a reasonable follow-up enhancement, not required for the
    MVP.

    Timing correction (see align.py/config.py) deliberately accepts a
    segment's synthesized audio running a bit long past its original slot
    rather than distorting the voice further to force an exact fit. Without
    a cap here, that overrun got summed directly on top of the *next*
    segment's audio starting at its own timestamp — two lines audibly
    overlapping instead of just drifting quietly out of sync (a real,
    reported issue). Each segment's audio is now truncated (with a short
    fade-out, not a hard click) at the next segment's start time if it
    would otherwise run into it.
    """
    background, bg_sr = sf.read(str(background_path), dtype="float32", always_2d=False)
    if background.ndim > 1:
        background = background.mean(axis=1)
    if bg_sr != sample_rate:
        background = librosa.resample(background, orig_sr=bg_sr, target_sr=sample_rate)

    total_len = max(
        len(background),
        max((int(s.end * sample_rate) for s in segments), default=0),
    )
    vocal_track = np.zeros(total_len, dtype=np.float32)

    ordered = sorted(segments, key=lambda s: s.start)
    fade_len = max(1, int(fade_ms / 1000 * sample_rate))
    for i, seg in enumerate(ordered):
        if seg.synthesized_audio is None or len(seg.synthesized_audio) == 0:
            continue
        start_sample = int(seg.start * sample_rate)
        audio = seg.synthesized_audio

        next_start_sample = int(ordered[i + 1].start * sample_rate) if i + 1 < len(ordered) else None
        if next_start_sample is not None and start_sample + len(audio) > next_start_sample:
            keep = max(0, next_start_sample - start_sample)
            if keep == 0:
                continue  # this segment's slot is already fully consumed by the next one
            audio = audio[:keep].copy()
            if len(audio) > fade_len:
                audio[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

        end_sample = start_sample + len(audio)
        if end_sample > len(vocal_track):
            vocal_track = np.pad(vocal_track, (0, end_sample - len(vocal_track)))
        vocal_track[start_sample:end_sample] += audio

    if len(background) < len(vocal_track):
        background = np.pad(background, (0, len(vocal_track) - len(background)))
    else:
        vocal_track = np.pad(vocal_track, (0, len(background) - len(vocal_track)))

    mixed = vocal_track + background
    peak = float(np.abs(mixed).max()) if len(mixed) else 0.0
    if peak > 0.98:
        log.warning("mixed output peak %.3f exceeds 0.98; scaling down to avoid clipping", peak)
        mixed = mixed * (0.98 / peak)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), mixed, sample_rate)
    return out_path
