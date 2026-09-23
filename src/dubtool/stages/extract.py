"""Video -> audio extraction (ffmpeg)."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


class FfmpegNotFoundError(RuntimeError):
    pass


def extract_audio(video_path: Path, out_path: Path, sample_rate: int = 44100) -> Path:
    """Pulls the audio track out of a video as stereo PCM WAV via ffmpeg.

    Stereo + 44.1kHz matches Demucs' native training configuration; keeping
    it stereo here (rather than downmixing to mono) matters for separation
    quality. Downstream stages downmix/resample to whatever *they* need.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "2", "-ar", str(sample_rate),
        "-acodec", "pcm_s16le", str(out_path),
    ]
    log.debug("running: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise FfmpegNotFoundError(
            "ffmpeg was not found on PATH. Install it (e.g. `brew install ffmpeg`)."
        ) from e
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed extracting audio from {video_path}:\n{result.stderr}")
    return out_path


def has_video_stream(path: Path) -> bool:
    """Some inputs are audio-only despite a video-like extension (e.g. a
    .mp4 container holding just an AAC track) — this is a real, legitimate
    case (podcasts, extracted audio tracks, ...), not just a malformed file,
    so the pipeline needs to know upfront whether there's a video stream to
    mux the dubbed audio back into at all.
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v",
        "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed inspecting {path}:\n{result.stderr}")
    return bool(result.stdout.strip())
