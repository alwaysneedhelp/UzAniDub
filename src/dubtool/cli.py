from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dubtool.config import DubConfig
from dubtool.pipeline import run as run_pipeline
from dubtool.registry import build_backends


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dubtool", description="Dub a video into Uzbek, cloning each speaker's voice."
    )
    p.add_argument("video", type=Path, help="input video file")
    p.add_argument("--target", default="uz", help="target language code (default: uz)")
    p.add_argument(
        "--output", "-o", type=Path, default=None,
        help="output path (default: output/<input filename>, overwritten on each run — "
             "pass an explicit path if you want to keep multiple runs side by side)",
    )
    p.add_argument("--config", type=Path, help="YAML config file overriding defaults")
    p.add_argument(
        "--num-speakers", type=int, default=None,
        help="hint the diarizer with a known speaker count (ignored by the step-2 single-speaker stub)",
    )
    p.add_argument(
        "--keep-intermediate", action="store_true",
        help="keep extracted audio, separated tracks, etc. in the work dir for debugging",
    )
    p.add_argument(
        "--emotion", default=None,
        help="force a TTS delivery style preset instead of cloning the reference speaker's own "
             "prosody (default: clone their natural prosody — see docs/config.py for why)",
    )
    p.add_argument(
        "--names", default=None,
        help="comma-separated proper nouns (character/place names, etc.) to bias transcription "
             "toward and protect from mistranslation — matters most for names a generic ASR/MT "
             "model has never seen, e.g. anime character names",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("dubtool.cli")

    config = DubConfig.from_yaml(args.config) if args.config else DubConfig()
    config.target_language = args.target
    if args.num_speakers is not None:
        config.num_speakers = args.num_speakers
    config.keep_intermediate = args.keep_intermediate
    if args.emotion:
        config.tts_emotion = args.emotion
    if args.names:
        config.proper_nouns = [name.strip() for name in args.names.split(",") if name.strip()]

    output_dir = Path("output")
    output_path = args.output or (output_dir / args.video.name)
    if config.keep_intermediate and config.work_dir is None:
        # Predictable, alongside the output, instead of a randomly-named
        # system temp dir — nothing useful is gained by hiding this path
        # from the user when they've explicitly asked to keep it.
        config.work_dir = output_dir / f"{args.video.stem}_work"

    backends = build_backends(config)
    try:
        result = run_pipeline(args.video, output_path, config, backends)
    except Exception as e:
        log.error("dubbing failed: %s", e)
        return 1

    print(f"Wrote {result.output_video}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
