"""Rewrites common English slang/idiomatic phrases into plainer English
before translation.

MADLAD-400 (like most classic NMT models — this isn't MADLAD-specific) is
trained overwhelmingly on formal/web-crawled parallel text and handles
literal, plain English far more reliably than casual slang — verified on
real content: "for real" translated into something unrelated to what was
actually meant, while a plain rephrasing ("really") translates correctly.

This is a cheap, fast fix (regex phrase substitution, no model call
involved) rather than swapping the whole translation backend for a much
heavier instruction-tuned LLM: on this project's CPU-only pipeline, an
LLM good enough to reliably handle slang would add a large per-segment
latency cost, and Uzbek specifically is a low-resource language most open
LLMs still handle poorly (real benchmarks show smaller open models
producing incoherent or language-mixed output on Uzbek) — so a generic
"bigger model" swap isn't a guaranteed win here the way it might be for a
high-resource target language.

English-only and deliberately not exhaustive: only applied when the
segment's source language is English (the phrases below have no meaning
in another language's transcript), and limited to phrases with one
dominant plain-English meaning across contexts — a genuinely ambiguous
phrase (e.g. "no way", which can mean either disbelief or literal
impossibility depending on context) is deliberately left out rather than
risk rewriting it wrong. Add more entries as new mistranslations turn up.
"""
from __future__ import annotations

import re

# Longest phrases first so e.g. "for real for real" isn't partially
# clobbered by the shorter "for real" pattern matching inside it first.
_REPLACEMENTS: list[tuple[str, str]] = [
    ("for real for real", "truly"),
    ("not gonna lie", "honestly"),
    ("hits different", "feels different"),
    ("big yikes", "how embarrassing"),
    ("vibe check", "mood check"),
    ("hard pass", "definitely not"),
    ("say less", "understood"),
    ("on god", "I swear"),
    ("my bad", "sorry, my mistake"),
    ("no cap", "no lie"),
    ("for real", "really"),
    ("lowkey", "somewhat"),
    ("highkey", "definitely"),
    ("deadass", "honestly"),
    ("rizz", "charisma"),
]

_PATTERNS = [
    (re.compile(r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b", re.IGNORECASE), plain)
    for phrase, plain in _REPLACEMENTS
]


def normalize(text: str, source_lang: str | None) -> str:
    if (source_lang or "").split("-")[0].lower() != "en":
        return text
    for pattern, plain in _PATTERNS:
        text = pattern.sub(plain, text)
    return text
