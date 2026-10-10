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
import re

log = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """Translate the following {source_lang} text into natural, everyday {target_lang}, \
the way a fluent native speaker would actually say it out loud in casual conversation \
or a dubbed video — not a stiff, literal, or overly formal translation. Preserve slang, \
idioms, tone, and register as best you can (translate their *meaning/feeling*, not a \
word-for-word gloss). Output ONLY the translated text itself — no quotes, no notes, no \
explanations, nothing else.

Text:
{text}"""

# Appended when the pipeline knows how long the line may run for. Uzbek
# renders consistently longer than English here: on a real clip 13 of 16
# segments overran their slot, needing 1.3x more speaking time than the
# original dialogue allowed and up to 3x on the worst line. align.py won't
# compress past 0.95x (heavier compression is what makes a voice sound
# chipmunky), so mix.py truncates the overrun mid-word and fades it -- which
# is what made the dub sound chopped rather than flat. A seq2seq model has no
# way to be asked for concision; an instruction-following one does, so the
# fix belongs here rather than in the aligner.
_LENGTH_HINT = """

LENGTH TARGET: this line is dubbed over video, so aim for something speakable \
in about {seconds:.1f} seconds — roughly {words} words at a normal speaking pace. \
Prefer the more concise of two equally accurate phrasings.

Accuracy outranks length. Never drop, shorten or approximate part of the meaning \
to hit the target, and never substitute a vaguer word for a precise one. If the \
faithful rendering runs somewhat longer than the target, give the faithful one — \
running slightly long is fine, saying the wrong thing is not."""


class GeminiTranslator:
    # The prompt above asks for slang and idiom to be carried across by
    # meaning, so pipeline should hand this backend the real line rather
    # than stages/idioms.py's plainer rewrite of it.
    handles_idioms = True

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

        # Thinking is on by default for the 2.5 models, and on a prompt this
        # short its reasoning trace comes back *inside* response.text --
        # observed on real output: "THINK: The user wants a natural, everyday
        # Uzbek translation..." followed by four numbered candidates, all of
        # which got handed to the TTS and synthesized as speech (17s of audio
        # for a one-second line). There is nothing to reason about here, so
        # budget zero: it removes the failure instead of parsing around it.
        from google.genai import types
        self._config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        target_duration: float | None = None,
    ) -> str:
        if not text.strip():
            return ""
        prompt = _PROMPT_TEMPLATE.format(
            source_lang=_language_name(source_lang),
            target_lang=_language_name(target_lang),
            text=text,
        )
        if target_duration and target_duration > 0:
            # ~2.5 words/second is a normal dub pace; floor at 2 words so a
            # very short slot still asks for something sayable.
            words = max(2, int(target_duration * 2.5))
            prompt += _LENGTH_HINT.format(seconds=target_duration, words=words)
        response = self._client.models.generate_content(
            model=self._model_name, contents=prompt, config=self._config,
        )
        return _clean((response.text or "").strip(), text)


# Markers that mean the model explained itself instead of answering. Each is
# matched case-insensitively at the start of a line.
_COMMENTARY_PREFIXES = (
    "think:", "thinking:", "note:", "notes:", "explanation:", "translation:",
    "here is", "here's", "option", "literal:", "alternatively",
)


def _looks_like_commentary(line: str) -> bool:
    stripped = line.strip()
    # A trailing colon is a preamble introducing what follows, never a
    # translation. This catches the open-ended phrasings a prefix list can't
    # anticipate ("Common Uzbek idioms for this:") -- which is the whole
    # weakness of matching against a fixed vocabulary of openers.
    if stripped.endswith(":"):
        return True
    low = stripped.lower().lstrip("*#->0123456789. \t")
    return any(low.startswith(prefix) for prefix in _COMMENTARY_PREFIXES)


def _strip_decoration(line: str) -> str:
    line = line.strip()
    # list numbering: "1. foo" / "2) foo"
    line = re.sub(r"^[*#>\-\s]*\d+\s*[.)]\s*", "", line).strip()
    # When the model offers candidates it bolds the answer and leaves its
    # reasoning unbolded beside it, so the bold span *is* the translation.
    bold = re.search(r"\*\*(.+?)\*\*", line)
    if bold:
        line = bold.group(1).strip()
    else:
        line = line.lstrip("*#->").strip().strip("*").strip()
    # Trailing gloss the prompt didn't ask for: "Salom. (Literal: hello)"
    line = re.sub(r"\s*\((?:literal|lit|meaning|note|more)\b[^)]*\)\s*$", "", line, flags=re.IGNORECASE).strip()
    if len(line) >= 2 and line[0] == line[-1] and line[0] in "\"'":
        line = line[1:-1].strip()
    return line


def _clean(raw: str, source_text: str) -> str:
    """Reduce a model response to the bare translation.

    Even with thinking disabled, an instruction-following model can ignore
    "output only the translation" -- wrapping quotes, a markdown fence, a
    preamble, or several candidates to choose from. None of that is
    detectable downstream: synthesize() will read whatever it is aloud, so a
    response that isn't a translation becomes a line of narration in the
    finished dub. Fail loudly here instead.
    """
    if not raw:
        return ""

    # ```...``` fences
    fenced = re.match(r"^```[a-zA-Z]*\n(.*?)\n?```$", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()

    lines = [ln for ln in (l.strip() for l in raw.splitlines()) if ln]
    candidates = [
        _strip_decoration(ln) for ln in lines if not _looks_like_commentary(ln)
    ]
    candidates = [c for c in candidates if c]

    if not candidates:
        log.warning(
            "gemini returned no usable translation for %r (got %r); leaving the "
            "segment untranslated rather than synthesizing commentary",
            source_text[:60], raw[:120],
        )
        return ""

    cleaned = candidates[0]
    if len(lines) > 1 or cleaned != raw:
        log.warning(
            "gemini response needed cleanup for %r: kept %r from a %d-line reply",
            source_text[:60], cleaned[:80], len(lines),
        )

    # A translation runs maybe 2x the source at worst across these language
    # pairs; far beyond that means prose about the translation, not the
    # translation. Better to drop it than to speak it.
    if len(cleaned) > max(80, len(source_text) * 6):
        log.warning(
            "gemini reply for %r is %d chars against a %d-char source — "
            "discarding it as commentary rather than speaking it",
            source_text[:60], len(cleaned), len(source_text),
        )
        return ""
    return cleaned


_LANGUAGE_NAMES = {
    "en": "English", "ru": "Russian", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "es": "Spanish", "fr": "French", "de": "German",
    "ar": "Arabic", "tr": "Turkish", "uz": "Uzbek", "auto": "the source language",
}


def _language_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code, code)
