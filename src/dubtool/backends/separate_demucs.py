"""Demucs (MIT) vocal/background separation."""
from __future__ import annotations

import logging
from pathlib import Path

import soundfile as sf

log = logging.getLogger(__name__)


class DemucsSeparator:
    def __init__(self, model_name: str = "htdemucs"):
        from demucs.api import Separator

        log.info("loading Demucs model %s", model_name)
        self._separator = Separator(model=model_name, device="cpu")

    def separate(self, audio_path: Path, out_dir: Path) -> tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info("separating %s with Demucs (this can take a while on CPU)", audio_path)
        _, stems = self._separator.separate_audio_file(Path(audio_path))

        vocals = stems["vocals"].cpu().numpy()
        background = sum(t.cpu().numpy() for name, t in stems.items() if name != "vocals")

        vocals_path = out_dir / "vocals.wav"
        background_path = out_dir / "background.wav"
        sr = self._separator.samplerate
        sf.write(str(vocals_path), vocals.T, sr)
        sf.write(str(background_path), background.T, sr)
        return vocals_path, background_path
