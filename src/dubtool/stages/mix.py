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
) -> Path:
    """Places each segment's aligned synthesized audio at its original
    timestamp on a silent timeline, then sums with the separated background
    track (resampled/downmixed to `sample_rate`, the TTS output rate, which
    is the pipeline's working rate for the final mix).

    Step 2 downmixes background to mono for simplicity; preserving stereo
    background is a reasonable follow-up enhancement, not required for the
    MVP.
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

    for seg in segments:
        if seg.synthesized_audio is None or len(seg.synthesized_audio) == 0:
            continue
        start_sample = int(seg.start * sample_rate)
        end_sample = start_sample + len(seg.synthesized_audio)
        if end_sample > len(vocal_track):
            vocal_track = np.pad(vocal_track, (0, end_sample - len(vocal_track)))
        vocal_track[start_sample:end_sample] += seg.synthesized_audio

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
