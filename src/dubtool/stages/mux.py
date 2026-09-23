"""Final audio -> video muxing (ffmpeg)."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def mux(video_path: Path, audio_path: Path, out_path: Path) -> Path:
    """Replaces the original video's audio track with the dubbed one via
    ffmpeg, copying the video stream untouched (no re-encode)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac",
        "-shortest",
        str(out_path),
    ]
    log.debug("running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed muxing {audio_path} into {video_path}:\n{result.stderr}")
    return out_path


_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def resolve_audio_only_output(output_path: Path) -> Path:
    """The caller may have asked for a video container (`--output out.mp4`)
    without knowing the input has no video stream. We can't honestly write
    that file, so redirect to a same-named .wav sibling instead of silently
    producing a broken/empty video."""
    if output_path.suffix.lower() in _VIDEO_EXTENSIONS:
        return output_path.with_suffix(".wav")
    return output_path


def write_audio_only(final_audio_path: Path, output_path: Path) -> Path:
    """Used instead of mux() when the source has no video stream to mux
    into — the dubbed audio *is* the deliverable."""
    resolved = resolve_audio_only_output(output_path)
    if resolved != output_path:
        log.warning(
            "%s has no video stream to mux into; writing dubbed audio to %s instead of %s",
            final_audio_path, resolved, output_path,
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(final_audio_path), str(resolved)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing audio-only output to {resolved}:\n{result.stderr}")
    return resolved
