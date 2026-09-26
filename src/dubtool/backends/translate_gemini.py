"""Opt-in translator: Gemini, via the Google Gen AI API.

Not the default — reachable only via `backends.translate: gemini` in config.
Two real reasons it isn't: it's a paid, closed, cloud API call per segment
(network-dependent, sends the source dialogue to Google, costs money per
run — unlike every default backend in this project, which is a local,
permissively-licensed model), and it needs its own account/API key. What
it buys over MADLAD-400 (the default): MADLAD is a classic NMT model
trained on formal/web-crawled parallel text and mistranslates casual
slang it rarely saw in training (see stages/idioms.py, added as a cheap
partial fix for exactly this); an instruction-following model like Gemini
can be asked directly for a natural, idiomatic rendering — including
slang — instead of a literal one, without the local-CPU-latency or
Uzbek-specific-quality concerns a locally-hosted LLM would carry on this
project's CPU-only pipeline (see the conversation that led to
stages/idioms.py for that reasoning; those concerns don't apply to a
cloud API call, which is why this is a reasonable opt-in despite that
earlier conclusion).

Requires a Gemini API key: set the GEMINI_API_KEY environment variable
(e.g. in a .env file: GEMINI_API_KEY=...), get one at
https://aistudio.google.com/apikey.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """Translate the following {source_lang} text into natural, everyday {target_lang}, \
the way a fluent native speaker would actually say it out loud in casual conversation \
or a dubbed video — not a stiff, literal, or overly formal translation. Preserve slang, \
idioms, tone, and register as best you can (translate their *meaning/feeling*, not a \
word-for-word gloss). Output ONLY the translated text itself — no quotes, no notes, no \
explanations, nothing else.

Text:
{text}"""


class GeminiTranslator:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ):
        from google import genai

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "the gemini translator requires a Gemini API key. Set the GEMINI_API_KEY "
                "environment variable (e.g. in a .env file: GEMINI_API_KEY=...) — get one "
                "at https://aistudio.google.com/apikey."
            )
        self._model_name = model_name
        self._client = genai.Client(api_key=key)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text.strip():
            return ""
        prompt = _PROMPT_TEMPLATE.format(
            source_lang=_language_name(source_lang),
            target_lang=_language_name(target_lang),
            text=text,
        )
        response = self._client.models.generate_content(model=self._model_name, contents=prompt)
        translated = (response.text or "").strip()
        # A model that ignores the "no quotes" instruction is a known failure
        # mode for instruction-following LLMs on short single-line prompts —
        # strip one layer of wrapping quotes rather than feed a literally
        # quoted string into the rest of the pipeline.
        if len(translated) >= 2 and translated[0] == translated[-1] and translated[0] in "\"'":
            translated = translated[1:-1].strip()
        return translated


_LANGUAGE_NAMES = {
    "en": "English", "ru": "Russian", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "es": "Spanish", "fr": "French", "de": "German",
    "ar": "Arabic", "tr": "Turkish", "uz": "Uzbek", "auto": "the source language",
}


def _language_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code, code)
