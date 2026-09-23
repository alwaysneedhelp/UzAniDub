"""Protocols each pipeline stage is written against.

`pipeline.py` only ever talks to these shapes, never to a concrete library
(Demucs, faster-whisper, NLLB, CosyVoice2, ...) directly. That's what lets a
backend be swapped via config without touching orchestration code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from dubtool.types import Segment


class Separator(Protocol):
    """Splits an audio file into an isolated-vocals track and a background track."""

    def separate(self, audio_path: Path, out_dir: Path) -> tuple[Path, Path]:
        """Returns (vocals_path, background_path)."""
        ...


class Diarizer(Protocol):
    """Finds who's speaking when. The step-2 MVP uses a stub that returns one
    speaker spanning the whole clip; step 4 swaps in pyannote.audio here
    without changing anything downstream."""

    def diarize(self, audio_path: Path, num_speakers: int | None = None) -> list[Segment]:
        """Returns Segments with start/end/speaker_id populated (nothing else)."""
        ...


class Transcriber(Protocol):
    """Fills in text + word-level timestamps for each diarized segment."""

    def transcribe(self, audio_path: Path, segments: list[Segment]) -> list[Segment]:
        """Takes the Segments from diarize() and returns them (possibly
        re-split at ASR-detected boundaries within each speaker turn) with
        .text and .words populated. speaker_id is preserved/inherited."""
        ...


class Translator(Protocol):
    def translate(self, text: str, source_lang: str, target_lang: str) -> str: ...


class TTSBackend(Protocol):
    """Synthesizes speech in the target language, optionally cloning a voice
    from a short reference clip."""

    sample_rate: int

    def synthesize(
        self, text: str, reference_audio: Path, emotion: str = "calm", speed: float = 1.0
    ) -> np.ndarray:
        """Returns mono float32 audio at self.sample_rate.

        `speed` is a native playback-rate hint (1.0 = the model's normal
        pace, >1 faster, <1 slower) passed through to the backend's own
        synthesis-time speed control where it has one. This exists because
        pipeline.py needs most of a segment's duration correction to happen
        *before* the waveform-domain phase-vocoder stretch in align.py, not
        instead of it — a backend's native speed control (e.g. mel-spectrogram
        interpolation ahead of vocoding) is typically less artifact-prone
        than stretching a finished waveform after the fact, so align.py
        should only be doing small residual correction, not the bulk of the
        work. A backend with no native speed control can just ignore this
        and always return speed-1.0 audio — pipeline.py's residual
        phase-vocoder pass still gets it the rest of the way, at whatever
        quality cost that entails.
        """
        ...
