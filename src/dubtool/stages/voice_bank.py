"""Persistent, cross-run voice identity matching.

Without this, every `dubtool` run independently re-extracts its own
reference clip for each speaker from that file's own audio — reasonable
within one run, but it means the same character dubbed across separate
episodes/scenes isn't guaranteed to clone from the same reference each
time, since each run only ever sees its own file. A show needs the same
character to sound the same episode to episode.

This keeps a small on-disk directory of (audio, embedding, transcript,
quality score) entries, one per distinct voice recognized so far, shared
across runs. Matching is by voice-embedding similarity (Resemblyzer,
Apache-2.0), not by name — dubtool has no way to know a character's actual
name is "Naruto" from the audio alone. A new voice gets an
auto-incrementing generic label (speaker_001, speaker_002, ...); rename the
.wav/.json pair yourself for a friendlier label if you like — matching only
looks at the stored embedding, never the filename.

This only wraps the *speaker-level* fallback reference (used when a
segment's own audio isn't a usable reference on its own — see
stages/reference.py's extract_segment_reference) — per-segment references
stay untouched, so within-scene emotional variety and cross-episode
identity consistency both hold, just via two different mechanisms.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from dubtool.types import ReferenceClip

log = logging.getLogger(__name__)

_encoder = None  # lazy singleton — Resemblyzer's model load isn't free


def _get_encoder():
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        _encoder = VoiceEncoder()
    return _encoder


def _embed(audio_path: Path) -> np.ndarray:
    from resemblyzer import preprocess_wav
    wav = preprocess_wav(str(audio_path))
    return _get_encoder().embed_utterance(wav)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class VoiceBank:
    def __init__(self, directory: Path, similarity_threshold: float = 0.75):
        self.directory = directory
        self.similarity_threshold = similarity_threshold
        self.directory.mkdir(parents=True, exist_ok=True)

    def find_or_add(self, candidate_path: Path, candidate_text: str, candidate_score: float) -> ReferenceClip:
        """Returns the ReferenceClip to actually use: either a matched
        existing bank entry (updated in place if `candidate_score` beats
        its stored score), or a freshly registered new entry built from
        `candidate_path`/`candidate_text`.
        """
        try:
            candidate_embedding = _embed(candidate_path)
        except Exception as e:
            log.warning(
                "voice bank matching unavailable (%s: %s) — using this run's own "
                "extraction without cross-run consistency", type(e).__name__, e,
            )
            return ReferenceClip(path=candidate_path, start=0.0, end=0.0, text=candidate_text)

        best_id, best_similarity = None, -1.0
        for meta_path in self.directory.glob("*.json"):
            meta = json.loads(meta_path.read_text())
            similarity = _cosine_similarity(candidate_embedding, np.array(meta["embedding"]))
            if similarity > best_similarity:
                best_similarity, best_id = similarity, meta_path.stem

        if best_id is not None and best_similarity >= self.similarity_threshold:
            meta = json.loads((self.directory / f"{best_id}.json").read_text())
            if candidate_score > meta.get("score", float("-inf")):
                log.info(
                    "voice bank: this run's clip for %s scores better (%.3f > %.3f) — updating stored reference",
                    best_id, candidate_score, meta.get("score", float("-inf")),
                )
                self._save(best_id, candidate_path, candidate_text, candidate_embedding, candidate_score)
                return ReferenceClip(path=self.directory / f"{best_id}.wav", start=0.0, end=0.0, text=candidate_text)
            log.info("voice bank: matched existing voice %s (similarity=%.3f)", best_id, best_similarity)
            return ReferenceClip(path=self.directory / f"{best_id}.wav", start=0.0, end=0.0, text=meta.get("text", ""))

        new_id = self._next_id()
        log.info(
            "voice bank: no match found (best similarity=%.3f, need >=%.2f) — registering new voice %s",
            best_similarity, self.similarity_threshold, new_id,
        )
        self._save(new_id, candidate_path, candidate_text, candidate_embedding, candidate_score)
        return ReferenceClip(path=self.directory / f"{new_id}.wav", start=0.0, end=0.0, text=candidate_text)

    def _next_id(self) -> str:
        existing = []
        for p in self.directory.glob("speaker_*.json"):
            suffix = p.stem.rsplit("_", 1)[-1]
            if suffix.isdigit():
                existing.append(int(suffix))
        return f"speaker_{(max(existing) + 1) if existing else 1:03d}"

    def _save(self, id_: str, audio_path: Path, text: str, embedding: np.ndarray, score: float) -> None:
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        sf.write(str(self.directory / f"{id_}.wav"), data, sr)
        (self.directory / f"{id_}.json").write_text(
            json.dumps({"embedding": embedding.tolist(), "text": text, "score": score})
        )
