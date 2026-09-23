"""Step-2 stub: treats the whole clip as one speaker.

Step 4 replaces SingleSpeakerStub with a pyannote.audio-backed diarizer
behind the same Diarizer protocol (see interfaces.py) — nothing in
pipeline.py or registry.py's call sites needs to change for that swap.
"""
from __future__ import annotations

from pathlib import Path

import soundfile as sf

from dubtool.types import Segment


class SingleSpeakerStub:
    def diarize(self, audio_path: Path, num_speakers: int | None = None) -> list[Segment]:
        info = sf.info(str(audio_path))
        return [Segment(start=0.0, end=info.duration, speaker_id="SPEAKER_00")]
