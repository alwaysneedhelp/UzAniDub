"""CosyVoice2-0.5B (Apache-2.0) + aisha-org/navoiy-tts (Apache-2.0) Uzbek
fine-tune. Refactored from the validated poc/poc_infer.py — the only real
change from that script is that the model is constructed once here (in
__init__) and reused across synthesize() calls, since model construction
dominates per-call cost far more than a single synthesize() call does.
"""
from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _unwrap_state_dict(value):
    if not isinstance(value, dict):
        return value
    for key in ("state_dict", "model", "llm"):
        nested = value.get(key)
        if isinstance(nested, dict):
            return _unwrap_state_dict(nested)
    return value


class CosyVoiceNavoiyTTS:
    sample_rate = 24000

    def __init__(
        self,
        cosyvoice_source_dir: Path,
        base_model_dir: Path,
        checkpoint_path: Path,
        seed: int = 1986,
    ):
        cosyvoice_source_dir = Path(cosyvoice_source_dir).resolve()
        checkpoint_path = Path(checkpoint_path).resolve()
        sys.path.insert(0, str(cosyvoice_source_dir))
        sys.path.insert(0, str(cosyvoice_source_dir / "third_party" / "Matcha-TTS"))
        sys.path.insert(0, str(checkpoint_path.parent))

        import soundfile as sf
        import torch
        import torchaudio

        # torchaudio's default backend on this build routes through
        # TorchCodec, whose bundled dylib has a broken @rpath into the
        # Homebrew ffmpeg install (dlopen fails, no LC_RPATH). Bypass it
        # entirely with a plain soundfile-based load/save (see
        # poc/poc_infer.py, where this was first diagnosed).
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

        self._normalize = normalize
        self._torch = torch

        emotions_path = checkpoint_path.parent / "emotions_40h.json"
        entries = json.loads(emotions_path.read_text(encoding="utf-8"))
        self._emotions: dict[str, dict] = {}
        for entry in entries:
            self._emotions[entry["uz"].lower()] = entry
            for tag in entry["tag"].replace("[", " ").replace("]", " ").split():
                self._emotions[tag.lower()] = entry

        random.seed(seed)
        torch.manual_seed(seed)

        log.info("loading CosyVoice2 base model from %s", base_model_dir)
        self._model = CosyVoice2(
            str(Path(base_model_dir).resolve()), load_jit=False, load_trt=False, fp16=False
        )

        log.info("loading navoiy-tts checkpoint from %s", checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = _unwrap_state_dict(state)
        incompatible = self._model.model.llm.load_state_dict(state, strict=False)
        if incompatible.missing_keys:
            log.warning("%d missing checkpoint keys", len(incompatible.missing_keys))
        if incompatible.unexpected_keys:
            log.warning("%d unexpected checkpoint keys", len(incompatible.unexpected_keys))
        self._model.model.llm.eval()

    def synthesize(
        self, text: str, reference_audio: Path, emotion: str = "calm", speed: float = 1.0
    ) -> np.ndarray:
        entry = self._emotions.get(emotion.lower().strip("[]"))
        if entry is None:
            valid = ", ".join(sorted(self._emotions))
            raise ValueError(f"Unknown TTS emotion {emotion!r}. Available: {valid}")

        normalized_text = self._normalize(text, mode="infer")
        instruction = entry["instruct"].strip() + "<|endofprompt|>"

        # CosyVoice2's own `speed` is applied via mel-spectrogram
        # interpolation *before* vocoding (see cosyvoice/cli/model.py:
        # tts_mel = F.interpolate(tts_mel, size=int(shape/speed), ...)),
        # which is generally less artifact-prone than stretching a finished
        # waveform after the fact — that's why pipeline.py routes most of
        # its duration correction through here rather than through
        # stages/align.py's waveform-domain phase vocoder.
        chunks = []
        with self._torch.inference_mode():
            for result in self._model.inference_instruct2(
                normalized_text, instruction, str(Path(reference_audio).resolve()),
                stream=False, speed=speed,
            ):
                chunks.append(result["tts_speech"].detach().cpu())
        if not chunks:
            raise RuntimeError(f"CosyVoice2 returned no audio for text: {text!r}")

        audio = self._torch.cat(chunks, dim=1)
        return audio.squeeze(0).numpy().astype(np.float32)
