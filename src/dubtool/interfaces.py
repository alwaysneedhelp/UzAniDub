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

    def transcribe(
        self, audio_path: Path, segments: list[Segment], extra_proper_nouns: list[str] | None = None
    ) -> list[Segment]:
        """Takes the Segments from diarize() and returns them (possibly
        re-split at ASR-detected boundaries within each speaker turn) with
        .text and .words populated. speaker_id is preserved/inherited.

        `extra_proper_nouns` adds to whatever proper nouns the backend was
        already configured with, for a given call only — pipeline.py uses
        this to run a second transcription pass biased toward names
        auto-detected from the first pass, without needing a second
        backend instance. A backend without a proper-noun bias mechanism
        can just ignore it.
        """
        ...


class Translator(Protocol):
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        target_duration: float | None = None,
    ) -> str: ...
    # A backend that sets `handles_names = True` also accepts
    # translate(..., names=[...]): auto-detected names as hints, rather than
    # pipeline pinning them to their source spelling via the glossary.


class TTSBackend(Protocol):
    """Synthesizes speech in the target language, optionally cloning a voice
    from a short reference clip."""

    # False for the classic NMT backends (MADLAD/OPUS-MT/NLLB), which are
    # trained on formal parallel text and read slang literally -- pipeline
    # runs stages/idioms.py over their input to compensate. True for an
    # instruction-following backend that is *asked* for idiomatic output:
    # pre-flattening its input would throw away the register it is there to
    # preserve, and replacement text chosen to survive a literal model is
    # strictly worse input than the original line.
    handles_idioms: bool = False

    sample_rate: int

    def synthesize(
        self,
        text: str,
        reference_audio: Path,
        reference_text: str = "",
        emotion: str | None = None,
        speed: float = 1.0,
    ) -> np.ndarray:
        """Returns mono float32 audio at self.sample_rate.

        `reference_text` is the transcript of `reference_audio` — a backend
        capable of true zero-shot cloning (matching the reference speaker's
        own prosody/energy rather than imposing a fixed delivery style) needs
        this to align against. Pass "" if unknown; a backend without a
        zero-shot mode can just ignore it.

        `emotion`, when given, requests a specific delivery style
        (implementation-defined preset names) *instead of* cloning the
        reference's natural prosody — use this for deliberate style control,
        not as the default. Real testing showed always forcing an emotion
        (e.g. a "calm" instruction on every segment) flattens out a speaker's
        actual intonation and reads as robotic — leave it None to let
        zero-shot prosody cloning (via reference_text) carry through instead.

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
