"""Enforces a consistent, pre-decided Uzbek rendering for known terms and
names instead of leaving them to the translation model's own judgment call
each time.

Two real problems this solves at once:
  1. A general-purpose MT model has never seen most anime character names
     or show-specific terminology and will sometimes mistranslate or garble
     them (verified on real content: "Almighty Push!" became "Resurrect!").
  2. Even when a term *should* be translated (not just preserved verbatim —
     e.g. "Cursed Energy" has a real Uzbek rendering, unlike a character's
     name), a generic MT model has no memory across calls and can render
     the same term two different ways in two different segments. That's
     the more important case to solve for actual terminology, as opposed to
     names: nothing here happens to notice "Cursed Energy" got translated
     to two different phrases in two different lines, because each
     translate() call is independent — the fix is forcing the same output
     for the same input, not detecting drift after the fact.

`glossary` is a plain {source_term: target_translation} dict. A character
name that should just pass through unchanged is simply an entry mapping to
itself (e.g. {"Yahiko": "Yahiko"}) — protecting a name and enforcing a
term's fixed translation are the same mechanism, just with source==target
for the name case. This only enforces entries it was actually given — it
can't discover terms it wasn't told about.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)


def term_pattern(term: str) -> re.Pattern[str]:
    """A glossary term as a whole word or phrase, plus an English plural.

    Matching anywhere used to find "Curse" inside "Cursed" and leave the
    remainder glued to the fixed rendering ("La'natd"), and to turn "Cursed
    Spirits" into "La'nat ruhis". Lookarounds rather than \\b so a term that
    starts or ends with punctuation still matches; a possessive's apostrophe
    isn't a word character, so "Gojo's" still protects "Gojo"."""
    return re.compile(rf"(?<!\w){re.escape(term)}(?:e?s)?(?!\w)", re.IGNORECASE)


def protect(text: str, glossary: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Returns (text with known terms replaced by placeholders,
    {placeholder: fixed_target_translation}) — `restore()` below puts the
    *glossary's* translation back, not necessarily the original source text.
    """
    mapping: dict[str, str] = {}
    protected = text
    # Longest term first. In file order, "Cursed Technique" matched inside
    # "Reverse Cursed Technique" and consumed it, so the longer entry never
    # applied and MADLAD rendered "Reverse <term>" as "returning to <term>"
    # -- observed on real output. A term that contains another must claim
    # its text before the shorter one can.
    ordered = sorted(glossary.items(), key=lambda item: len(item[0]), reverse=True)
    for i, (term, translation) in enumerate(ordered):
        if not term.strip():
            continue
        placeholder = f"XPRESERVEDTERM{i}X"
        pattern = term_pattern(term)
        if pattern.search(protected):
            protected = pattern.sub(placeholder, protected)
            mapping[placeholder] = translation
    return protected, mapping


# A placeholder as the model gave it back, possibly with its letters
# mangled. Only the digits are relied on: an instruction-following backend
# treats XPRESERVEDTERM40X as text to be rewritten and will happily return
# XPRSERVEDTERM40X (one letter short), at which point an exact replace
# leaves the sentinel in the finished line and the TTS reads it aloud as
# gibberish. Observed on real Gemini output. The classic NMT backends pass
# the token through untouched, which is why this only shows up with an LLM.
_MANGLED_PLACEHOLDER = re.compile(r"X[A-Za-z]{0,24}?(\d{1,4})X")
_APOSTROPHE_AFTER_PLACEHOLDER = re.compile(r"(X[A-Za-z]{0,24}?\d{1,4}X)['ʼ’`](?=[^\W\d_])")


def restore(text: str, mapping: dict[str, str]) -> str:
    """Swaps placeholders back for their fixed glossary translations after
    translation.

    Matches on the index rather than the literal placeholder, so a backend
    that rewrote the surrounding letters still gets its term restored."""
    # An LLM sets a suffix off from a token with an apostrophe
    # ("XPRESERVEDTERM3X'ni"), which restored as "Teskarisi'ni" on real
    # Gemini output. No Uzbek suffix starts with one, and the term's own
    # apostrophes (Bo'sh) live inside its rendering, not after the token.
    text = _APOSTROPHE_AFTER_PLACEHOLDER.sub(r"\1", text)
    for placeholder, translation in mapping.items():
        text = text.replace(placeholder, translation)
    if not mapping or "X" not in text:
        return text

    by_index = {
        match.group(1): translation
        for placeholder, translation in mapping.items()
        if (match := re.fullmatch(r"X[A-Za-z]*?(\d{1,4})X", placeholder))
    }

    def _rescue(match: re.Match[str]) -> str:
        translation = by_index.get(match.group(1))
        if translation is None:
            return match.group(0)
        log.warning(
            "glossary placeholder came back altered as %r; restored %r by index",
            match.group(0), translation,
        )
        return translation

    return _MANGLED_PLACEHOLDER.sub(_rescue, text)
