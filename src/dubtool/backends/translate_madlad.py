"""Default translator: CTranslate2 int8 quantization of google/madlad400-3b-mt
(Apache-2.0), via zenoverflow/madlad400-3b-mt-int8-float32.

MADLAD-400 handles any source language in one hop (it infers the source
language itself; only a `<2xx>` target-language tag is needed), so this
backend ignores `source_lang` entirely — no two-hop pivot needed like the
OPUS-MT fallback in translate_opus_mt.py.

Why this over the vanilla fp32 checkpoint: google/madlad400-3b-mt's only
safetensors checkpoint is 11.76GB, which doesn't fit this machine's disk
budget alongside Demucs/faster-whisper/CosyVoice2. This int8 CTranslate2
quantization is ~3GB and reuses a dependency (ctranslate2) already
installed for faster-whisper — no new heavy dependency, much better
quality than the OPUS-MT fallback (see translate_opus_mt.py's docstring
for why that one is rough: real-world testing showed it dropping large
chunks of meaning and occasionally hallucinating unrelated phrases).
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class MadladTranslator:
    def __init__(self, model_name: str = "zenoverflow/madlad400-3b-mt-int8-float32", device: str = "cpu"):
        self._model_name = model_name
        self._device = device
        self._translator = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._translator is not None:
            return
        import ctranslate2
        import transformers
        from huggingface_hub import snapshot_download

        log.info("loading MADLAD-400 (CTranslate2 int8) from %s", self._model_name)
        model_path = snapshot_download(self._model_name)
        self._translator = ctranslate2.Translator(model_path, device=self._device)
        self._tokenizer = transformers.T5Tokenizer.from_pretrained(model_path)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text.strip():
            return ""
        self._ensure_loaded()

        input_text = f"<2{target_lang}> {text}"
        input_tokens = self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(input_text))
        results = self._translator.translate_batch([input_tokens])
        output_tokens = results[0].hypotheses[0]
        output_ids = self._tokenizer.convert_tokens_to_ids(output_tokens)
        return self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()
