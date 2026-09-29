"""The custom model hook loads one backend instance from a configured factory."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

from dubtool.backends.tts_custom import build_custom_tts
from dubtool.config import DubConfig


class CustomTTSTests(unittest.TestCase):
    def test_factory_receives_config_and_returns_backend(self):
        module = types.ModuleType("test_voice_plugin")
        seen = []

        class Backend:
            sample_rate = 24000

            def synthesize(self, text, reference_audio: Path, reference_text="", emotion=None, speed=1.0):
                return None

        def build(config):
            seen.append(config)
            return Backend()

        module.build = build
        sys.modules[module.__name__] = module
        try:
            config = DubConfig()
            config.models.custom_tts_factory = "test_voice_plugin:build"
            backend = build_custom_tts(config)
            self.assertEqual(backend.sample_rate, 24000)
            self.assertEqual(seen, [config])
        finally:
            del sys.modules[module.__name__]

    def test_missing_factory_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "custom_tts_factory"):
            build_custom_tts(DubConfig())


if __name__ == "__main__":
    unittest.main()
