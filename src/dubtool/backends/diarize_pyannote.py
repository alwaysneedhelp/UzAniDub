"""Real multi-speaker diarization via pyannote.audio.

Requires a Hugging Face access token with the gated models' license terms
accepted on both:
  https://huggingface.co/pyannote/speaker-diarization-3.1
  https://huggingface.co/pyannote/segmentation-3.0
(the diarization pipeline depends on the segmentation model internally).
Set the token via the HF_TOKEN environment variable (e.g. in a .env file)
or pass it directly.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dubtool.types import Segment

log = logging.getLogger(__name__)


class PyannoteDiarizer:
    def __init__(
        self,
        hf_token: str | None = None,
        model_name: str = "pyannote/speaker-diarization-3.1",
    ):
        from pyannote.audio import Pipeline

        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "pyannote diarization requires a Hugging Face access token. Set the "
                "HF_TOKEN environment variable (e.g. in a .env file: HF_TOKEN=hf_...) "
                "after accepting the gated model terms at "
                "huggingface.co/pyannote/speaker-diarization-3.1 and "
                "huggingface.co/pyannote/segmentation-3.0."
            )
        log.info("loading pyannote diarization pipeline: %s", model_name)
        pipeline = Pipeline.from_pretrained(model_name, token=token)
        if pipeline is None:
            raise RuntimeError(
                f"pyannote.audio failed to load {model_name!r} — check that the token is "
                "valid and both gated models' terms have been accepted."
            )
        self._pipeline = pipeline

    def diarize(self, audio_path: Path, num_speakers: int | None = None) -> list[Segment]:
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers

        diarization = self._pipeline(str(audio_path), **kwargs)

        segments = [
            Segment(start=float(turn.start), end=float(turn.end), speaker_id=str(speaker))
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        segments.sort(key=lambda s: s.start)

        found = len({s.speaker_id for s in segments})
        log.info("diarization found %d speaker(s) across %d turn(s)", found, len(segments))
        return segments
