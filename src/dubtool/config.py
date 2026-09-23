"""Config schema + YAML loading.

Every model choice, path, and pacing knob lives here so a user can override
one thing (say, swap the translation backend) without editing code. CLI flags
override config-file values, which override these defaults.
"""
from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelPaths:
    # Uzbek voice-cloning TTS stack validated in poc/ (CosyVoice2-0.5B base +
    # aisha-org/navoiy-tts Apache-2.0 Uzbek fine-tune). Paths are relative to
    # the project root unless absolute.
    cosyvoice_source_dir: Path = Path("poc/CosyVoice")
    cosyvoice_base_dir: Path = Path("poc/models/CosyVoice2-0.5B")
    navoiy_checkpoint: Path = Path("poc/models/navoiy-tts/emotion_600h_joint.pt")
    default_reference_audio: Path = Path("assets/default_reference.wav")
    # Exact transcript of default_reference_audio (generated via macOS `say`
    # — see poc/reference.wav) — known up front since we synthesized it
    # ourselves, letting the bundled fallback voice use zero-shot prosody
    # cloning too instead of falling further back to a flat instruct2 default.
    default_reference_text: str = (
        "Hi, this is a short sample recording used only to test a voice "
        "cloning research prototype. It is not a real person's voice."
    )

    whisper_model: str = "small"
    # Default translator: an int8 CTranslate2 quantization of MADLAD-400
    # (google/madlad400-3b-mt, Apache-2.0) via zenoverflow's ~3GB conversion
    # (reuses ctranslate2, already a dependency for faster-whisper). The
    # original plan was the vanilla fp32 checkpoint, but that's an 11.76GB
    # download that didn't fit this machine's disk budget — the interim
    # OPUS-MT fallback (still available, see below) turned out too lossy in
    # real testing (dropped large chunks of meaning, occasional
    # hallucination) to keep as the default. See THIRD_PARTY_LICENSES.md.
    madlad_ct2_model: str = "zenoverflow/madlad400-3b-mt-int8-float32"
    # Non-default alternatives, still wired up behind the same Translator
    # protocol (dubtool.interfaces) and selectable via backends["translate"]:
    opus_mt_en_trk_model: str = "Helsinki-NLP/opus-mt-en-trk"
    opus_mt_mul_en_model: str = "Helsinki-NLP/opus-mt-mul-en"
    nllb_model: str = "facebook/nllb-200-distilled-600M"  # CC-BY-NC-4.0, opt-in only
    demucs_model: str = "htdemucs"


@dataclass
class DubConfig:
    target_language: str = "uz"
    source_language: str | None = None  # None = auto-detect at transcription time

    # Hint for pyannote (ignored by the single-speaker stub).
    num_speakers: int | None = None
    # Hugging Face access token for pyannote's gated diarization models —
    # None reads from the HF_TOKEN environment variable instead (see
    # backends/diarize_pyannote.py). Exists as a config field mainly so a
    # YAML config can set it without relying on env var setup.
    hf_token: str | None = None

    # Proper nouns (character/place names, etc.) that matter for both
    # transcription accuracy and translation fidelity — see
    # transcribe_whisper.py (passed as an ASR initial_prompt to bias
    # recognition) and stages/proper_nouns.py (kept verbatim through
    # translation rather than risking mistranslation/mangling by the MT
    # model). Empty by default; set via `--names` or a config file. Matters
    # most for content with names an ASR/MT model has never seen — e.g.
    # anime character names — where a generic model will otherwise
    # transcribe/translate them into the nearest common word.
    proper_nouns: list[str] = field(default_factory=list)

    keep_intermediate: bool = False
    # None => pipeline.run() allocates a fresh temp dir per run. Different
    # invocations used to default to the same ".dubtool_work" path and
    # silently clobber/mix each other's intermediate files (a real bug hit
    # while debugging real output) — leave this unset unless you specifically
    # want a fixed, predictable location.
    work_dir: Path | None = None

    # Timing correction happens in two stages now, not one — real testing
    # showed a single waveform-domain phase-vocoder pass (align.py alone)
    # sounded "robotic" and "too stretched" even after max_stretch_ratio was
    # already lowered from 2.0 to 1.25, because that one stage was carrying
    # the *entire* duration-correction burden:
    #
    #   1. pipeline.py synthesizes each segment once at speed=1.0 to measure
    #      its natural duration, then re-synthesizes at a computed native
    #      `speed` (clamped to [min_native_speed, max_native_speed]) using
    #      the TTS backend's own synthesis-time speed control where it has
    #      one (e.g. mel-spectrogram interpolation ahead of vocoding) —
    #      this does most of the work and is typically much less
    #      artifact-prone than stretching a finished waveform.
    #   2. align.py's waveform-domain phase-vocoder stretch then only needs
    #      to close whatever small gap remains, bounded tightly by
    #      [min_stretch_ratio, max_stretch_ratio] — outside that range we
    #      accept timing drift (the dubbed segment ends early/late relative
    #      to the original) rather than distort the voice further.
    #
    # min/max_stretch_ratio are therefore much tighter than before (used to
    # bound the *entire* correction; now only the small residual after step
    # 1).
    #
    # Both sets of bounds were tightened further after real anime-dialogue
    # testing still sounded "stretchy" even under the two-pass scheme: 4 of
    # 5 segments in that run hit a clamp, and with the old bounds
    # (0.7-1.5 native, 0.9-1.1 residual) a segment needing heavy correction
    # could still end up ~1.5x*1.1x ≈ 1.65x off natural pace in the worst
    # case — both stages maxed out simultaneously, compounding their
    # artifacts. Dramatic/expressive delivery (anime dialogue, unlike calm
    # narration) makes even a "modest" 10-50% stretch read as audibly
    # unnatural. Tightened to prioritize accepting timing drift over
    # distorting the voice much more aggressively than before — this is a
    # real tradeoff, not a free lunch: expect more segments to end early/
    # late relative to the original now.
    min_native_speed: float = 0.85
    max_native_speed: float = 1.2
    min_stretch_ratio: float = 0.95
    max_stretch_ratio: float = 1.05

    # None (the default) means: don't force a delivery style, clone the
    # reference speaker's own natural prosody instead (see
    # backends/tts_cosyvoice.py and interfaces.py's TTSBackend.synthesize
    # docstring for why — real testing showed forcing "calm" on every
    # segment flattened the source speaker's actual intonation and read as
    # robotic). Set explicitly (e.g. via --emotion) only when deliberate
    # style control is wanted over prosody fidelity.
    tts_emotion: str | None = None

    backends: dict[str, str] = field(
        # "translate" defaults to madlad (Apache-2.0, ~3GB int8 CTranslate2
        # quantization of MADLAD-400) — not NLLB-200 (CC-BY-NC-4.0,
        # non-commercial only, opt-in via backends["translate"] = "nllb",
        # see THIRD_PARTY_LICENSES.md), and not opus_mt (Apache-2.0 and
        # still available via backends["translate"] = "opus_mt", but real
        # testing showed it dropping large chunks of meaning and
        # occasionally hallucinating unrelated phrases — madlad is both
        # better-licensed-than-NLLB and meaningfully better quality than
        # opus_mt).
        # "diarize" defaults to "auto": real multi-speaker diarization
        # (pyannote) if an HF_TOKEN is available (env var or config), else
        # the single-speaker stub — pyannote's models are gated and need a
        # token with their license terms accepted, so this avoids a hard
        # dependency on that setup for anyone just trying the tool out.
        # Force one explicitly via backends["diarize"] = "pyannote" or
        # "single_speaker_stub" to skip the auto-detection.
        default_factory=lambda: {
            "translate": "madlad",
            "tts": "cosyvoice_navoiy",
            "transcribe": "faster_whisper",
            "diarize": "auto",
            "separate": "demucs",
        }
    )

    models: ModelPaths = field(default_factory=ModelPaths)

    @classmethod
    def from_yaml(cls, path: Path) -> "DubConfig":
        raw = yaml.safe_load(path.read_text()) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> "DubConfig":
        raw = dict(raw)
        models_raw = raw.pop("models", {})
        cfg = cls(**{k: v for k, v in raw.items() if k != "models"})
        if models_raw:
            cfg.models = ModelPaths(**models_raw)
        _coerce_paths(cfg)
        _coerce_paths(cfg.models)
        return cfg


_PATH_TYPE_RE = re.compile(r"\bPath\b")


def _coerce_paths(obj: Any) -> None:
    """YAML gives us strings; fields typed as Path should actually be Path.

    Word-boundary match matters here: a naive substring check ("Path" in
    str(f.type)) also matches "ModelPaths", which would wrongly try to
    coerce that whole nested dataclass into a pathlib.Path.
    """
    if not is_dataclass(obj):
        return
    for f in fields(obj):
        if _PATH_TYPE_RE.search(str(f.type)):
            value = getattr(obj, f.name)
            if value is not None and not isinstance(value, Path):
                setattr(obj, f.name, Path(value))
