"""faster-whisper transcription with word-level timestamps."""
from __future__ import annotations

import logging
from pathlib import Path

from dubtool.types import Segment, Word

log = logging.getLogger(__name__)


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        language: str | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
        proper_nouns: list[str] | None = None,
    ):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._language = language  # None = auto-detect
        # Whisper's initial_prompt biases decoding toward vocabulary it's
        # seen recently in the "context" — feeding known proper nouns here
        # measurably helps recognize names an ASR model would otherwise
        # guess at (e.g. transcribing an unfamiliar name as the nearest
        # common word/acronym it does know). Matters most for content like
        # anime with character names no general-purpose model has seen.
        self._initial_prompt = (
            "Character and place names in this audio: " + ", ".join(proper_nouns) + "."
            if proper_nouns
            else None
        )

    def transcribe(self, audio_path: Path, segments: list[Segment]) -> list[Segment]:
        whisper_segments, info = self._model.transcribe(
            str(audio_path),
            language=self._language,
            word_timestamps=True,
            initial_prompt=self._initial_prompt,
        )
        speaker_id = segments[0].speaker_id if segments else "SPEAKER_00"

        result: list[Segment] = []
        for ws in whisper_segments:
            words = [Word(text=w.word.strip(), start=w.start, end=w.end) for w in (ws.words or [])]
            result.append(
                Segment(
                    start=ws.start,
                    end=ws.end,
                    speaker_id=speaker_id,
                    text=ws.text.strip(),
                    words=words,
                    detected_language=info.language,
                )
            )
        log.info(
            "transcribed %d segment(s), detected language=%s (p=%.2f)",
            len(result), info.language, info.language_probability,
        )
        return result
