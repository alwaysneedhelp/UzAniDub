"""How registry picks (and falls back on) a diarization backend.

A token is not the same as access: pyannote's models are gated
individually, and pyannote.audio 4.x additionally needs
speaker-diarization-community-1, whose gate is manually approved rather
than instant. So "HF_TOKEN is set" cannot be treated as "pyannote will
load" -- before the fallback below, setting a token up to *enable*
diarization turned a working single-speaker run into one that died at
stage 3, which is strictly worse than having no token at all.

The asymmetry these tests pin: "auto" degrades, an explicit "pyannote"
raises. Someone who named the backend wants to know it didn't load, not
to find a one-voice dub hours later.
"""
from __future__ import annotations

import pytest

from dubtool.backends.diarize_resemblyzer import ResemblyzerDiarizer
from dubtool.config import DubConfig
from dubtool.registry import _build_diarizer
from dubtool.stages.diarize import SingleSpeakerStub


class Gated(Exception):
    """Stands in for huggingface_hub's gated-repo 403."""


@pytest.fixture
def no_env_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)


def _fail_pyannote(monkeypatch, exc=None):
    import dubtool.backends.diarize_pyannote as mod

    def boom(*a, **k):
        raise exc or Gated("403 Client Error. Cannot access gated repo")
    monkeypatch.setattr(mod, "PyannoteDiarizer", boom)


def test_no_token_selects_resemblyzer_without_touching_pyannote(no_env_token):
    # No token can't mean one voice for a whole episode when an ungated
    # multi-speaker diarizer is already installed.
    config = DubConfig()
    config.backends["diarize"] = "auto"
    assert isinstance(_build_diarizer(config), ResemblyzerDiarizer)


def test_auto_falls_back_to_resemblyzer_when_pyannote_is_gated(monkeypatch, no_env_token):
    monkeypatch.setenv("HF_TOKEN", "hf_pretend")
    _fail_pyannote(monkeypatch)
    config = DubConfig()
    config.backends["diarize"] = "auto"
    assert isinstance(_build_diarizer(config), ResemblyzerDiarizer)


def test_auto_fallback_warns_loudly_enough_to_notice(monkeypatch, no_env_token, caplog):
    monkeypatch.setenv("HF_TOKEN", "hf_pretend")
    _fail_pyannote(monkeypatch)
    config = DubConfig()
    config.backends["diarize"] = "auto"
    with caplog.at_level("WARNING"):
        _build_diarizer(config)
    assert any(r.levelname == "WARNING" for r in caplog.records)
    assert "resemblyzer" in caplog.text


def test_explicit_pyannote_raises_instead_of_degrading(monkeypatch, no_env_token):
    # The whole point of naming it: don't silently hand back one voice.
    monkeypatch.setenv("HF_TOKEN", "hf_pretend")
    _fail_pyannote(monkeypatch)
    config = DubConfig()
    config.backends["diarize"] = "pyannote"
    with pytest.raises(Gated):
        _build_diarizer(config)


def test_explicit_stub_is_honoured_even_with_a_token(monkeypatch, no_env_token):
    # Still reachable on request: it's the only backend that costs nothing
    # when you already know the source is one speaker.
    monkeypatch.setenv("HF_TOKEN", "hf_pretend")
    config = DubConfig()
    config.backends["diarize"] = "single_speaker_stub"
    assert isinstance(_build_diarizer(config), SingleSpeakerStub)


def test_explicit_resemblyzer_never_consults_a_token(monkeypatch, no_env_token):
    monkeypatch.setenv("HF_TOKEN", "hf_pretend")
    config = DubConfig()
    config.backends["diarize"] = "resemblyzer"
    assert isinstance(_build_diarizer(config), ResemblyzerDiarizer)


def test_unknown_name_is_rejected(no_env_token):
    config = DubConfig()
    config.backends["diarize"] = "nonsense"
    with pytest.raises(ValueError, match="unknown diarize backend"):
        _build_diarizer(config)
