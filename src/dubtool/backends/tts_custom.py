"""Load a user-provided TTS backend once per dubbing run.

The factory path is configured in models.custom_tts_factory. Keeping the model
instance alive avoids reloading a large checkpoint for every speech segment.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dubtool.config import DubConfig
    from dubtool.interfaces import TTSBackend


def build_custom_tts(config: DubConfig) -> TTSBackend:
    spec = config.models.custom_tts_factory
    if not spec or ":" not in spec:
        raise ValueError(
            "custom TTS requires models.custom_tts_factory: "
            "'your_module:build_tts'"
        )
    module_name, factory_name = spec.rsplit(":", 1)
    if not module_name or not factory_name:
        raise ValueError(f"invalid custom TTS factory: {spec!r}")
    factory = getattr(importlib.import_module(module_name), factory_name)
    backend = factory(config)
    if not isinstance(getattr(backend, "sample_rate", None), int) or backend.sample_rate <= 0:
        raise TypeError("custom TTS backend must expose a positive integer sample_rate")
    if not callable(getattr(backend, "synthesize", None)):
        raise TypeError("custom TTS backend must implement synthesize()")
    return backend
