"""Maps config backend-name strings to concrete implementations.

Imports for each concrete backend (Demucs/faster-whisper/MADLAD/CosyVoice)
are lazy, inside each branch, so something like `dubtool --help` — or a unit
test that only needs align.py — doesn't pay their import cost or require
them installed.

build_backends() hands pipeline.Backends the *factories* rather than
constructed instances, so each model is loaded when its stage starts and
freed when that stage ends (see pipeline.Backends for why). preflight()
exists to keep the fail-fast behaviour that deferring construction would
otherwise cost: without it, a bad backend name or a missing TTS
checkpoint wouldn't surface until Stage 7, half an hour of Demucs and
Whisper into a run.
"""
from __future__ import annotations

import logging
import os

from dubtool.config import DubConfig
from dubtool.pipeline import Backends

log = logging.getLogger(__name__)

_KNOWN_BACKENDS = {
    "separate": {"demucs"},
    "diarize": {"auto", "pyannote", "resemblyzer", "single_speaker_stub"},
    "transcribe": {"faster_whisper"},
    "translate": {"madlad", "opus_mt", "nllb", "gemini"},
    "tts": {"cosyvoice_navoiy"},
}


def build_backends(config: DubConfig) -> Backends:
    return Backends(
        separator=lambda: _build_separator(config),
        diarizer=lambda: _build_diarizer(config),
        transcriber=lambda: _build_transcriber(config),
        translator=lambda: _build_translator(config),
        tts=lambda: _build_tts(config),
    )


def preflight(config: DubConfig) -> None:
    """Check what can be checked without loading a single weight: that every
    configured backend name is one we implement, and that the TTS stack's
    local files are actually on disk. Everything else (MADLAD, Whisper,
    Demucs, pyannote) downloads itself on first use, so its absence isn't an
    error we can detect here — but CosyVoice2 and navoiy-tts are
    hand-installed under poc/models, and a missing checkpoint there is the
    failure most worth catching before Demucs spends ten minutes."""
    for stage, name in config.backends.items():
        known = _KNOWN_BACKENDS.get(stage)
        if known is not None and name not in known:
            raise ValueError(
                f"unknown {stage} backend: {name!r} (known: {sorted(known)})"
            )

    if config.backends.get("tts") == "cosyvoice_navoiy":
        required = {
            "CosyVoice source checkout": config.models.cosyvoice_source_dir,
            "CosyVoice2 base model": config.models.cosyvoice_base_dir,
            "navoiy-tts checkpoint": config.models.navoiy_checkpoint,
        }
        missing = [f"{label} ({path})" for label, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "TTS backend 'cosyvoice_navoiy' is missing required local files:\n  - "
                + "\n  - ".join(missing)
                + "\nSee Setup.md / setup_openvoice.sh, or point config.models at them."
            )

    if not config.models.default_reference_audio.exists():
        # Not fatal: only segments with no usable cloning reference of their
        # own fall back to it, which may never happen in a given run.
        log.warning(
            "bundled fallback reference voice is missing (%s); segments with no "
            "usable reference of their own will fail to synthesize",
            config.models.default_reference_audio,
        )


def _build_separator(config: DubConfig):
    name = config.backends["separate"]
    if name == "demucs":
        from dubtool.backends.separate_demucs import DemucsSeparator
        return DemucsSeparator(model_name=config.models.demucs_model)
    raise ValueError(f"unknown separate backend: {name!r}")


def _single_speaker_stub():
    from dubtool.stages.diarize import SingleSpeakerStub
    return SingleSpeakerStub()


def _resemblyzer_diarizer(config: DubConfig):
    from dubtool.backends.diarize_resemblyzer import ResemblyzerDiarizer
    return ResemblyzerDiarizer()


def _build_diarizer(config: DubConfig):
    name = config.backends["diarize"]
    explicit = name != "auto"
    if name == "auto":
        name = "pyannote" if (config.hf_token or os.environ.get("HF_TOKEN")) else "resemblyzer"
    if name == "resemblyzer":
        return _resemblyzer_diarizer(config)
    if name == "pyannote":
        from dubtool.backends.diarize_pyannote import PyannoteDiarizer
        try:
            return PyannoteDiarizer(hf_token=config.hf_token)
        except Exception as e:
            # "auto" means "use the best diarizer actually available", and
            # having a token is not the same as having access: pyannote's
            # models are individually gated, and pyannote.audio 4.x pulls a
            # PLDA transform from speaker-diarization-community-1, whose gate
            # is manually approved rather than instant like 3.1's and
            # segmentation-3.0's. Before this, setting up a token to *enable*
            # diarization turned a working single-speaker run into a run that
            # died at stage 3 -- strictly worse than having no token at all.
            # Someone who asked for pyannote by name still gets the error.
            if explicit:
                raise
            log.warning(
                "pyannote diarization is unavailable (%s: %s); falling back to the "
                "resemblyzer diarizer, which needs no gated models. If that was a "
                "gated-access error, request access to the model it names "
                "(pyannote.audio %s requires it) and re-run to get pyannote's "
                "better boundaries and overlap handling.",
                type(e).__name__, str(e).splitlines()[0][:200],
                _pyannote_version(),
            )
            return _resemblyzer_diarizer(config)
    if name == "single_speaker_stub":
        return _single_speaker_stub()
    raise ValueError(f"unknown diarize backend: {name!r}")


def _pyannote_version() -> str:
    try:
        from importlib.metadata import version
        return version("pyannote.audio")
    except Exception:
        return "unknown"


def _build_transcriber(config: DubConfig):
    name = config.backends["transcribe"]
    if name == "faster_whisper":
        from dubtool.backends.transcribe_whisper import FasterWhisperTranscriber
        return FasterWhisperTranscriber(
            model_size=config.models.whisper_model,
            language=config.source_language,
            proper_nouns=list(config.glossary.keys()),
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
    if name == "gemini":
        # Paid, closed, cloud API — opt-in, not the default. See
        # backends/translate_gemini.py for why it's worth having anyway
        # (better slang/idiom handling than MADLAD) and THIRD_PARTY_LICENSES.md
        # before using it for anything beyond personal/research use.
        from dubtool.backends.translate_gemini import GeminiTranslator
        return GeminiTranslator(model_name=config.models.gemini_model)
    raise ValueError(f"unknown translate backend: {name!r}")


def _build_tts(config: DubConfig):
    name = config.backends["tts"]
    if name == "custom":
        from dubtool.backends.tts_custom import build_custom_tts

        return build_custom_tts(config)
    if name == "cosyvoice_navoiy":
        from dubtool.backends.tts_cosyvoice import CosyVoiceNavoiyTTS
        return CosyVoiceNavoiyTTS(
            cosyvoice_source_dir=config.models.cosyvoice_source_dir,
            base_model_dir=config.models.cosyvoice_base_dir,
            checkpoint_path=config.models.navoiy_checkpoint,
        )
    raise ValueError(f"unknown tts backend: {name!r}")
