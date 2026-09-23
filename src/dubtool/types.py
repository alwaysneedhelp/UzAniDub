"""Shared data types passed between pipeline stages.

Kept deliberately dumb (plain dataclasses, no behavior) so every stage module
can be tested by constructing these directly, without running the stage
before it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Word:
    text: str
    start: float  # seconds, absolute (relative to the full extracted audio)
    end: float


@dataclass
class ReferenceClip:
    """A voice-cloning reference clip for one speaker, plus what they
    actually said during it. The transcript matters for zero-shot cloning
    (see backends/tts_cosyvoice.py): CosyVoice2's inference_zero_shot clones
    the reference's own natural prosody, but needs to know the reference
    clip's text to align against — unlike inference_instruct2, which
    replaces natural prosody with a fixed instruction ("speak calmly") and
    so doesn't need it, at the cost of flattening the speaker's real
    delivery.
    """

    path: Path
    start: float
    end: float
    text: str = ""


@dataclass
class Segment:
    """One utterance turn: a single speaker talking between start and end.

    Populated incrementally as it flows through the pipeline:
    diarize -> (start, end, speaker_id)
    transcribe -> + text, words
    translate -> + translated_text
    synthesize -> + synthesized_audio, synthesized_sr
    align -> synthesized_audio replaced with a time-stretched version whose
             length matches (end - start)
    """

    start: float
    end: float
    speaker_id: str
    text: str = ""
    words: list[Word] = field(default_factory=list)
    detected_language: str | None = None  # set by the transcriber (e.g. whisper's language-ID), used to pick a translation pivot
    translated_text: str = ""
    reference_audio: Path | None = None
    reference_text: str = ""  # transcript of reference_audio, for zero-shot prosody cloning
    synthesized_audio: np.ndarray | None = None
    synthesized_sr: int | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DubbingResult:
    output_video: Path
    segments: list[Segment]
    intermediate_dir: Path | None = None
