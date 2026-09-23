"""Maps config backend-name strings to concrete implementations.

Imports for each concrete backend (Demucs/faster-whisper/MADLAD/CosyVoice)
are lazy, inside each branch, so something like `dubtool --help` — or a unit
test that only needs align.py — doesn't pay their import cost or require
them installed.
"""
from __future__ import annotations

import os

from dubtool.config import DubConfig
from dubtool.pipeline import Backends


def build_backends(config: DubConfig) -> Backends:
    return Backends(
        separator=_build_separator(config),
        diarizer=_build_diarizer(config),
        transcriber=_build_transcriber(config),
        translator=_build_translator(config),
        tts=_build_tts(config),
    )


def _build_separator(config: DubConfig):
    name = config.backends["separate"]
    if name == "demucs":
        from dubtool.backends.separate_demucs import DemucsSeparator
        return DemucsSeparator(model_name=config.models.demucs_model)
    raise ValueError(f"unknown separate backend: {name!r}")


def _build_diarizer(config: DubConfig):
    name = config.backends["diarize"]
    if name == "auto":
        name = "pyannote" if (config.hf_token or os.environ.get("HF_TOKEN")) else "single_speaker_stub"
    if name == "pyannote":
        from dubtool.backends.diarize_pyannote import PyannoteDiarizer
        return PyannoteDiarizer(hf_token=config.hf_token)
    if name == "single_speaker_stub":
        from dubtool.stages.diarize import SingleSpeakerStub
        return SingleSpeakerStub()
    raise ValueError(f"unknown diarize backend: {name!r}")


def _build_transcriber(config: DubConfig):
    name = config.backends["transcribe"]
    if name == "faster_whisper":
        from dubtool.backends.transcribe_whisper import FasterWhisperTranscriber
        return FasterWhisperTranscriber(
            model_size=config.models.whisper_model,
            language=config.source_language,
            proper_nouns=config.proper_nouns,
        )
    raise ValueError(f"unknown transcribe backend: {name!r}")


def _build_translator(config: DubConfig):
    name = config.backends["translate"]
    if name == "madlad":
        from dubtool.backends.translate_madlad import MadladTranslator
        return MadladTranslator(model_name=config.models.madlad_ct2_model)
    if name == "opus_mt":
        # Apache-2.0, much lighter (~600MB vs ~3GB), but meaningfully rougher
        # quality — see translate_opus_mt.py's docstring. Kept as a fallback
        # for constrained environments, not the default.
        from dubtool.backends.translate_opus_mt import OpusMtTranslator
        return OpusMtTranslator(
            en_trk_model=config.models.opus_mt_en_trk_model,
            mul_en_model=config.models.opus_mt_mul_en_model,
        )
    if name == "nllb":
        # CC-BY-NC-4.0 — non-commercial only. Opt-in, not the default.
        # See THIRD_PARTY_LICENSES.md before enabling this in anything
        # that isn't purely personal/research use.
        from dubtool.backends.translate_nllb import NLLBTranslator
        return NLLBTranslator(model_name=config.models.nllb_model)
    raise ValueError(f"unknown translate backend: {name!r}")


def _build_tts(config: DubConfig):
    name = config.backends["tts"]
    if name == "cosyvoice_navoiy":
        from dubtool.backends.tts_cosyvoice import CosyVoiceNavoiyTTS
        return CosyVoiceNavoiyTTS(
            cosyvoice_source_dir=config.models.cosyvoice_source_dir,
            base_model_dir=config.models.cosyvoice_base_dir,
            checkpoint_path=config.models.navoiy_checkpoint,
        )
    raise ValueError(f"unknown tts backend: {name!r}")
