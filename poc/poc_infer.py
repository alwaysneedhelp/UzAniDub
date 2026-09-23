#!/usr/bin/env python3
"""Run a Navoiy TTS LLM checkpoint with the upstream CosyVoice2 runtime.

Patched from aisha-org/navoiy-tts inference.py to run CPU-only (no CUDA
available on this machine). Changes from the original:
  1. Removed the "CUDA GPU is required" hard guard.
  2. fp16=True -> fp16=False (fp16 matmul is unsupported/slow on CPU).
  3. torch.cuda.manual_seed_all() only called if CUDA is actually available.
Everything else (state-dict swap, uztts normalize, inference_instruct2 call)
is unchanged from the original sanctioned integration path.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_EMOTIONS = ROOT / "models" / "navoiy-tts" / "emotions_40h.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosyvoice-dir", type=Path, help="CosyVoice source checkout")
    parser.add_argument("--base-model-dir", type=Path, help="Downloaded CosyVoice2-0.5B directory")
    parser.add_argument("--checkpoint", type=Path, help="Navoiy .pt LLM checkpoint")
    parser.add_argument("--reference", type=Path, help="Consented reference-speaker WAV")
    parser.add_argument("--text", help="Uzbek text to synthesize")
    parser.add_argument("--emotion", default="calm", help="Emotion preset name or tag")
    parser.add_argument("--emotions-file", type=Path, default=DEFAULT_EMOTIONS)
    parser.add_argument("--output", type=Path, default=Path("output.wav"))
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1986)
    parser.add_argument("--list-emotions", action="store_true")
    args = parser.parse_args()

    if not args.list_emotions:
        required = ("cosyvoice_dir", "base_model_dir", "checkpoint", "reference", "text")
        missing = [name.replace("_", "-") for name in required if not getattr(args, name)]
        if missing:
            parser.error("missing required arguments: " + ", ".join(f"--{name}" for name in missing))
    if args.speed <= 0:
        parser.error("--speed must be greater than zero")
    return args


def load_emotions(path: Path) -> tuple[list[dict], dict[str, dict]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[str, dict] = {}
    for entry in entries:
        lookup[entry["uz"].lower()] = entry
        for tag in entry["tag"].replace("[", " ").replace("]", " ").split():
            lookup[tag.lower()] = entry
    return entries, lookup


def unwrap_state_dict(value):
    if not isinstance(value, dict):
        return value
    for key in ("state_dict", "model", "llm"):
        nested = value.get(key)
        if isinstance(nested, dict):
            return unwrap_state_dict(nested)
    return value


def main() -> None:
    args = parse_args()
    entries, emotions = load_emotions(args.emotions_file)

    if args.list_emotions:
        for entry in entries:
            aliases = ", ".join(
                part.strip() for part in entry["tag"].replace("[", "").split("]") if part.strip()
            )
            print(f"{entry['uz']}: {aliases}")
        return

    emotion = emotions.get(args.emotion.lower().strip("[]"))
    if emotion is None:
        valid = ", ".join(sorted(emotions))
        raise SystemExit(f"Unknown emotion {args.emotion!r}. Available names/tags: {valid}")

    cosyvoice_dir = args.cosyvoice_dir.resolve()
    sys.path.insert(0, str(cosyvoice_dir))
    sys.path.insert(0, str(cosyvoice_dir / "third_party" / "Matcha-TTS"))
    sys.path.insert(0, str(args.checkpoint.resolve().parent))

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    # torchaudio's default backend on this build routes through TorchCodec,
    # whose bundled libtorchcodec_core*.dylib has a broken @rpath into the
    # Homebrew ffmpeg install (dlopen fails with "no LC_RPATH's found").
    # Bypass it entirely with a plain soundfile-based load/save.
    def _sf_load(filepath, frame_offset=0, num_frames=-1, normalize=True,
                 channels_first=True, format=None, backend=None, buffer_size=4096):
        data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T if channels_first else data)
        return wav, sr

    def _sf_save(filepath, src, sample_rate, channels_first=True, format=None,
                 encoding=None, bits_per_sample=None, buffer_size=4096, backend=None,
                 compression=None):
        arr = src.detach().cpu().numpy()
        if channels_first:
            arr = arr.T
        sf.write(str(filepath), arr, sample_rate)

    torchaudio.load = _sf_load
    torchaudio.save = _sf_save

    from cosyvoice.cli.cosyvoice import CosyVoice2
    from uztts.normalize import normalize

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available} (running on {'GPU' if cuda_available else 'CPU'})")

    for path, label in (
        (args.base_model_dir, "base model"),
        (args.checkpoint, "checkpoint"),
        (args.reference, "reference WAV"),
    ):
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda_available:
        torch.cuda.manual_seed_all(args.seed)

    model = CosyVoice2(
        str(args.base_model_dir.resolve()),
        load_jit=False,
        load_trt=False,
        fp16=False,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = unwrap_state_dict(state)
    incompatible = model.model.llm.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        print(f"Warning: {len(incompatible.missing_keys)} missing checkpoint keys", file=sys.stderr)
    if incompatible.unexpected_keys:
        print(f"Warning: {len(incompatible.unexpected_keys)} unexpected checkpoint keys", file=sys.stderr)
    model.model.llm.eval()

    text = normalize(args.text, mode="infer")
    instruction = emotion["instruct"].strip() + "<|endofprompt|>"
    chunks = []
    with torch.inference_mode():
        for result in model.inference_instruct2(
            text,
            instruction,
            str(args.reference.resolve()),
            stream=False,
            speed=args.speed,
        ):
            chunks.append(result["tts_speech"].detach().cpu())
    if not chunks:
        raise SystemExit("The model returned no audio.")

    audio = torch.cat(chunks, dim=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output), audio, 24000)
    duration = audio.shape[-1] / 24000
    print(f"Wrote {args.output} ({duration:.2f}s, emotion={args.emotion}, seed={args.seed})")


if __name__ == "__main__":
    main()
