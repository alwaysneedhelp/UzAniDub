"""Extracts a short, clean reference clip per speaker for voice cloning.

Diarization gives us a (possibly single, for the pre-step-4 MVP) speaker_id
and time ranges where they're speaking; from those, we pick a short
contiguous window of real speech to hand to the TTS backend as its
zero-shot cloning reference. This is a simple heuristic (sustained RMS
energy, no clipping, not too much internal silence), not a VAD model — good
enough to dodge the failure modes that actually matter (grabbing a silent
gap or a clipped/breath-noise moment), not a claim of picking the *best*
possible clip.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from dubtool.types import Segment

log = logging.getLogger(__name__)


def extract_reference_clips(
    vocals_path: Path,
    segments: list[Segment],
    out_dir: Path,
    target_duration: float = 8.0,
    min_duration: float = 4.0,
    hop: float = 1.0,
    raw_audio_path: Path | None = None,
    background_path: Path | None = None,
    background_energy_ratio: float = 0.1,
) -> dict[str, Path]:
    """One reference clip per distinct speaker_id in `segments`, written to
    `out_dir/reference_<speaker_id>.wav`. Returns {speaker_id: path}; a
    speaker with no window of at least `min_duration` clean seconds is
    omitted — the caller falls back to the bundled default reference voice
    for them.

    If `raw_audio_path` and `background_path` are both given and the
    background track's energy in the chosen window is low relative to the
    vocals there (meaning Demucs had little to actually remove for that
    stretch), the clip is pulled from the *raw, pre-separation* audio for
    that same time range instead of the separated vocals track.
    Source-separation models can leave a faint "processed" quality in their
    vocals stem even when there's nothing audible to remove, and since
    voice cloning imitates whatever timbre the reference clip has, that
    artifact gets baked into the *entire* dubbed output's voice — not just
    the reference moment. Using untouched raw audio whenever it's safe to
    (nothing meaningful was actually separated out there) avoids that.
    """
    data, sr = sf.read(str(vocals_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)

    raw_audio = None
    background = None
    if raw_audio_path is not None and background_path is not None:
        bg, bg_sr = sf.read(str(background_path), dtype="float32", always_2d=False)
        if bg.ndim > 1:
            bg = bg.mean(axis=1)
        raw, raw_sr = sf.read(str(raw_audio_path), dtype="float32", always_2d=False)
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        if bg_sr == sr and raw_sr == sr:
            background, raw_audio = bg, raw
        else:
            log.warning(
                "raw/background sample rates (%s/%s) don't match vocals (%d); "
                "skipping raw-audio substitution for the reference clip",
                raw_sr, bg_sr, sr,
            )

    by_speaker: dict[str, list[Segment]] = {}
    for seg in segments:
        by_speaker.setdefault(seg.speaker_id, []).append(seg)

    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for speaker_id, speaker_segments in by_speaker.items():
        window = _best_window(data, sr, speaker_segments, target_duration, min_duration, hop)
        if window is None:
            log.warning(
                "not enough clean speech to build a reference clip for %s; "
                "falling back to the default reference voice for them",
                speaker_id,
            )
            continue
        clip_data, start_sample = window

        if background is not None and raw_audio is not None:
            end_sample = start_sample + len(clip_data)
            if end_sample <= len(background) and end_sample <= len(raw_audio):
                vocal_rms = float(np.sqrt(np.mean(clip_data**2)))
                bg_rms = float(np.sqrt(np.mean(background[start_sample:end_sample] ** 2)))
                if vocal_rms > 0 and bg_rms < background_energy_ratio * vocal_rms:
                    log.info(
                        "using raw pre-separation audio for %s's reference clip "
                        "(background energy negligible in that window)", speaker_id,
                    )
                    clip_data = raw_audio[start_sample:end_sample]

        path = out_dir / f"reference_{speaker_id}.wav"
        sf.write(str(path), clip_data, sr)
        result[speaker_id] = path
    return result


def _best_window(
    data: np.ndarray,
    sr: int,
    speaker_segments: list[Segment],
    target_duration: float,
    min_duration: float,
    hop: float,
) -> tuple[np.ndarray, int] | None:
    win_len = int(target_duration * sr)
    hop_len = max(1, int(hop * sr))
    min_len = int(min_duration * sr)

    best_score = -1.0
    best_window: np.ndarray | None = None
    best_start_sample = 0

    for seg in speaker_segments:
        start_sample = max(0, int(seg.start * sr))
        end_sample = min(len(data), int(seg.end * sr))
        span = data[start_sample:end_sample]
        if len(span) < min_len:
            continue

        this_win_len = min(win_len, len(span))
        for offset in range(0, max(1, len(span) - this_win_len + 1), hop_len):
            window = span[offset : offset + this_win_len]
            if len(window) < min_len:
                continue
            rms = float(np.sqrt(np.mean(window**2)))
            peak = float(np.abs(window).max())
            if peak > 0.99:  # likely clipping — avoid handing the cloner a distorted reference
                continue
            silence_frac = float(np.mean(np.abs(window) < 0.01))
            score = rms * (1.0 - silence_frac)
            if score > best_score:
                best_score = score
                best_window = window
                best_start_sample = start_sample + offset

    if best_window is None:
        return None
    return best_window, best_start_sample
