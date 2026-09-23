"""Opt-in translator: NLLB-200-distilled-600M.

CC-BY-NC-4.0 — non-commercial only. Not the default (see config.py /
THIRD_PARTY_LICENSES.md); only reachable via `backends.translate: nllb` in
config for users who accept that restriction. Correctness has not been
validated end-to-end here — the default `opus_mt` backend is what's proven
against the integration test.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# whisper/ISO-639-1-ish codes -> NLLB FLORES-200 codes.
_FLORES_CODES = {
    "en": "eng_Latn", "ru": "rus_Cyrl", "es": "spa_Latn", "fr": "fra_Latn",
    "de": "deu_Latn", "ar": "arb_Arab", "zh": "zho_Hans", "tr": "tur_Latn",
    "uz": "uzn_Latn", "uzn": "uzn_Latn",
}


class NLLBTranslator:
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        log.info("loading NLLB translation model %s (CC-BY-NC-4.0, non-commercial only)", model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._model.eval()

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        import torch

        if not text.strip():
            return ""
        src = _FLORES_CODES.get(source_lang, source_lang)
        tgt = _FLORES_CODES.get(target_lang, target_lang)
        self._tokenizer.src_lang = src
        inputs = self._tokenizer(text, return_tensors="pt")
        forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(tgt)
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs, forced_bos_token_id=forced_bos_token_id, max_new_tokens=512
            )
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
