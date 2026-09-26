from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from dubtool.config import DubConfig
from dubtool.pipeline import run as run_pipeline
from dubtool.registry import build_backends


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dubtool", description="Dub a video into Uzbek."
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
        help="hint the diarizer with a known speaker count (used by pyannote; ignored by the "
             "single-speaker stub)",
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
        help="character/place names to bias transcription toward and preserve verbatim through "
             "translation — matters most for names a generic ASR/MT model has never seen, e.g. "
             "anime character names. Either a comma-separated list (\"Naruto,Sasuke,Sakura\") "
             "for a one-off, or a path to a text file (one name per line, blank lines and "
             "#-comments ignored) so a whole show's cast list can be reused across episodes "
             "without retyping it every run. Each name is added to the glossary as an identity "
             "entry (passes through unchanged) — for terms that should be *translated* to a "
             "fixed rendering instead, use --glossary",
    )
    p.add_argument(
        "--glossary", type=Path, default=None,
        help="path to a YAML file of {source term: fixed target translation} pairs, enforced "
             "consistently on every occurrence instead of leaving it to the translation model's "
             "own per-call judgment (which has no memory across segments and can render the "
             "same recurring term two different ways). A name that should pass through "
             "unchanged is just an entry mapping to itself. See assets/jjk_glossary.yaml for an "
             "example built for Jujutsu Kaisen",
    )
    p.add_argument(
        "--no-clone-voices", action="store_true",
        help="use the bundled generic Navoiy TTS reference voice for every segment instead of "
             "cloning each speaker's own voice from a reference clip extracted from the source "
             "audio (cloning is the default). Faster and simpler, at the cost of every speaker "
             "sounding the same",
    )
    p.add_argument(
        "--voice-bank-dir", type=Path, default=None,
        help="directory for the persistent cross-run voice bank (default: ./voice_bank) — "
             "lets the same character clone consistently across separate runs (e.g. different "
             "episodes of a show) instead of each run re-extracting independently. Only takes "
             "effect with voice cloning enabled (the default; see --no-clone-voices)",
    )
    p.add_argument(
        "--no-voice-bank", action="store_true",
        help="disable the persistent voice bank — this run's reference clips stay purely "
             "self-contained, matching the original (pre-voice-bank) behavior",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def parse_names(value: str) -> list[str]:
    """`--names` accepts either an inline comma-separated list or a path to
    a persistent cast-list file (one name per line; blank lines and lines
    starting with # ignored) — a whole show's characters only need typing
    out once, not on every single episode/scene run."""
    path = Path(value)
    if path.is_file():
        names = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
        return names
    return [name.strip() for name in value.split(",") if name.strip()]


def parse_glossary(path: Path) -> dict[str, str]:
    """Loads a {source term: fixed target translation} YAML file — see
    assets/jjk_glossary.yaml for the format."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping of {{term: translation}}, got {type(raw).__name__}")
    return {str(k): str(v) for k, v in raw.items()}


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()  # picks up e.g. HF_TOKEN from a .env file, if present

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
        for name in parse_names(args.names):
            config.glossary[name] = name
    if args.glossary:
        config.glossary.update(parse_glossary(args.glossary))
    if args.no_clone_voices:
        config.clone_voices = False
    if args.no_voice_bank:
        config.voice_bank_dir = None
    elif args.voice_bank_dir is not None:
        config.voice_bank_dir = args.voice_bank_dir

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
