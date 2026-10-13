"""Put a dubbed soundtrack back into its video.

For dubs made from an audio-only upload (e.g. the Kaggle notebook given a
.flac/.m4a instead of the episode itself): the video stream is copied
untouched and the dubbed audio replaces the original track, exactly as a
dub of the full video would have done it (stages/mux.py).

Usage:
    python scripts/combine_av.py episode.mp4 episode_uz.flac
    python scripts/combine_av.py episode.mkv episode_uz.flac -o output/episode_uz.mkv

The output defaults to the audio's name with the video's extension, next to
the audio: episode_uz.flac -> episode_uz.mp4.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dubtool.stages.mux import mux

# How far the two may differ before it looks like the wrong pair. The dub can
# run a little past the picture -- lines are never cut short -- but not by
# seconds; dub_batch.sh uses the same tolerance.
_MAX_DRIFT = 2.0


def _duration(path: Path) -> float | None:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Replace a video's audio with a dubbed soundtrack.")
    p.add_argument("video", type=Path, help="original video (its picture is kept)")
    p.add_argument("audio", type=Path, help="dubbed audio, e.g. episode_uz.flac")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="output video (default: the audio's name with the video's extension)")
    args = p.parse_args(argv)

    for path in (args.video, args.audio):
        if not path.is_file():
            p.error(f"no such file: {path}")
    out = args.output or args.audio.with_suffix(args.video.suffix)
    if out.resolve() == args.video.resolve():
        p.error(f"refusing to overwrite the source video {args.video}; pass -o")

    video_s, audio_s = _duration(args.video), _duration(args.audio)
    if video_s is not None and audio_s is not None and abs(video_s - audio_s) > _MAX_DRIFT:
        print(
            f"warning: video is {video_s:.1f}s but audio is {audio_s:.1f}s -- "
            "check this is the right pair", file=sys.stderr,
        )

    mux(args.video, args.audio, out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
