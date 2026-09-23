#!/usr/bin/env python3
"""Generates the fully-synthetic, copyright-clean sample video used by the
integration test: one clear English speaker (macOS `say`) over a quiet
background tone (so the separate() stage has real non-silent background
content to preserve), muxed onto a static-color video track.

Re-run this to regenerate tests/integration/fixtures/sample_video.mp4 if it
ever needs to change; the committed .mp4 is the output of this script, not a
hand-authored asset.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SPEECH_TEXT = (
    "Hello, my name is Alex. Today the weather is very nice, and I am going "
    "for a walk in the park. I hope you have a wonderful day."
)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    speech_aiff = FIXTURES_DIR / "_speech.aiff"
    speech_wav = FIXTURES_DIR / "_speech.wav"
    mixed_audio = FIXTURES_DIR / "_mixed_audio.wav"
    out_video = FIXTURES_DIR / "sample_video.mp4"

    run(["say", "-v", "Alex", SPEECH_TEXT, "-o", str(speech_aiff)])
    run(["ffmpeg", "-y", "-i", str(speech_aiff), "-ar", "44100", "-ac", "2", str(speech_wav)])

    duration = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(speech_wav)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    )
    duration = max(duration, 1.0)

    # Quiet background tone under the speech — low enough to not mask
    # intelligibility, present enough for Demucs to have real content to
    # separate and for a listener to notice if it's missing after dubbing.
    run([
        "ffmpeg", "-y",
        "-i", str(speech_wav),
        "-f", "lavfi", "-i", f"sine=frequency=110:duration={duration}",
        "-filter_complex", "[1:a]volume=0.06[bg];[0:a][bg]amix=inputs=2:duration=first[aout]",
        "-map", "[aout]", str(mixed_audio),
    ])

    run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=0x3b5b92:s=640x360:d={duration}",
        "-i", str(mixed_audio),
        "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p",
        str(out_video),
    ])

    for tmp in (speech_aiff, speech_wav, mixed_audio):
        tmp.unlink(missing_ok=True)

    print(f"\nWrote {out_video} ({duration:.1f}s)")


if __name__ == "__main__":
    sys.exit(main())
