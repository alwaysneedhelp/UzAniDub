"""Default translator: two Helsinki-NLP OPUS-MT models (Apache-2.0), pivoting
through English for any source language that isn't already English.

Why not a single multilingual model: MADLAD-400 (google/madlad400-3b-mt) was
the original plan and directly supports "any source language -> Uzbek" in
one hop, but its only safetensors checkpoint is 11.76GB, which doesn't fit
this machine's disk budget alongside Demucs/faster-whisper/CosyVoice2. Its
GGUF quantized variants (~1-2GB) would fit, but T5-architecture GGUF
inference support in transformers/llama.cpp is shaky enough that it wasn't
worth the integration risk for an MVP. OPUS-MT is ~300MB per model, uses the
well-supported MarianMT architecture, and is Apache-2.0.

Model choice:
- Helsinki-NLP/opus-mt-en-trk: English -> a group of Turkic languages,
  selected via a `>>id<<` target-language tag prefixed to the input text.
  Uzbek (`uzb_Latn` / `uzb_Cyrl`) is one of its trained targets. We always
  ask for `uzb_Latn` (Latin script) since that's what navoiy-tts's
  normalizer expects.
- Helsinki-NLP/opus-mt-mul-en: many-source-languages -> English, used only
  when the detected source language isn't already English, to avoid an
  unnecessary and lossy English->English round trip.

Quality caveat: this pair was not built for direct-to-Uzbek translation and
Uzbek is one of many low-resource targets opus-mt-en-trk was trained on
(reported eng-uzb BLEU ~3.4 in its own model card, versus ~35 for eng-tur) —
translations will be rough. This is a known, documented limitation of the
MVP's default backend, not a bug; MADLAD-400 or a fine-tuned model are
worth revisiting once GPU/disk budget allows.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_ENGLISH_CODES = {"en", "eng", "english"}

# ISO-639-1-ish codes (as faster-whisper/whisper report them) -> the
# uzb_Latn/uzb_Cyrl-style FLORES/Tatoeba codes opus-mt-en-trk expects for
# its *target* tag. We only ever target Uzbek here, so this is fixed.
_TARGET_TAG = {"uz": "uzb_Latn", "uzn": "uzb_Latn"}


class OpusMtTranslator:
    def __init__(self, en_trk_model: str = "Helsinki-NLP/opus-mt-en-trk",
                 mul_en_model: str = "Helsinki-NLP/opus-mt-mul-en"):
        self._en_trk_name = en_trk_model
        self._mul_en_name = mul_en_model
        self._en_trk = None  # lazy: always needed eventually, loaded on first use
        self._mul_en = None  # lazy: only needed for non-English sources

    def _load(self, model_name: str):
        from transformers import MarianMTModel, MarianTokenizer
        log.info("loading translation model %s", model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model.eval()
        return tokenizer, model

    def _en_trk_pair(self):
        if self._en_trk is None:
            self._en_trk = self._load(self._en_trk_name)
        return self._en_trk

    def _mul_en_pair(self):
        if self._mul_en is None:
            self._mul_en = self._load(self._mul_en_name)
        return self._mul_en

    def _generate(self, tokenizer, model, text: str) -> str:
        import torch
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=512)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text.strip():
            return ""

        target_tag = _TARGET_TAG.get(target_lang, target_lang)

        english_text = text
        normalized_source = (source_lang or "").lower()
        if normalized_source not in _ENGLISH_CODES:
            tokenizer, model = self._mul_en_pair()
            english_text = self._generate(tokenizer, model, text)
            log.debug("pivoted %r (%s) -> English: %r", text, source_lang, english_text)

        tokenizer, model = self._en_trk_pair()
        tagged = f">>{target_tag}<< {english_text}"
        return self._generate(tokenizer, model, tagged)
