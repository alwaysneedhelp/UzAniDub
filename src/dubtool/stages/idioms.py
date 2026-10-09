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

# Phrases that are only unambiguous as a *whole* utterance, so they can't
# go in _REPLACEMENTS without breaking honest inline uses. "Exactly!" on its
# own is agreement; "exactly five minutes" means precisely, and rewriting the
# latter would be a new bug. Matched against the entire segment with
# surrounding punctuation ignored.
#
# "Exactly!" is here because MADLAD rendered it as "Yo'q!" -- *"No!"* -- on
# real output: an agreement inverted into a refusal. A confident, fluent
# reversal of meaning is the worst failure mode this layer can prevent,
# since nothing downstream can detect it.
_STANDALONE: list[tuple[str, str]] = [
    ("exactly", "yes, that is correct"),
    ("precisely", "yes, that is correct"),
    ("no kidding", "that is surprising"),
    ("you bet", "yes, certainly"),
    ("damn right", "yes, certainly"),
    ("hell yeah", "yes, definitely"),
    ("not a chance", "that is impossible"),
    ("beats me", "I do not know"),
    ("got it", "I understand"),
    ("hang on", "wait"),
    ("hold up", "wait"),
    ("knock it off", "stop that"),
    ("shut it", "be quiet"),
    ("nice one", "well done"),
    ("my point exactly", "that is what I meant"),
]

# Inline idioms whose figurative sense is stable in any context. Each one
# below was either observed mistranslating on real output or is the same
# shape as one that did -- a figurative phrase MADLAD renders literally.
_FIGURATIVE: list[tuple[str, str]] = [
    # Observed: became "I brought it to the edge of death" -- subject and
    # meaning both scrambled by the figurative "verge".
    ("on the verge of death", "when I was almost dead"),
    ("on the verge of", "very close to"),
    # Observed twice. The literal gloss lost the idiom; the obvious
    # rewrite ("much harder than it sounds") then had "sounds" read as the
    # *noun* and came back as "more than his voice". So the replacement has
    # to avoid any word a literal NMT model can take for a noun.
    ("easier said than done", "that is very difficult to actually do"),
    # Observed: "Is he high?" became "Is he tall?" -- the drug sense is
    # dominant for a person, but bare "high" is far too ambiguous to touch,
    # so only these person-subject forms are rewritten.
    ("is he high", "is he on drugs"),
    ("is she high", "is she on drugs"),
    ("are you high", "are you on drugs"),
    ("he is high", "he is on drugs"),
    ("you are high", "you are on drugs"),
    ("piece of cake", "very easy"),
    ("no big deal", "not important"),
    ("take it easy", "relax"),
    ("in the nick of time", "just in time"),
    ("no sweat", "easily"),
    ("you are done for", "you will be defeated"),
    ("done for", "finished"),
    ("give it your all", "try with your full effort"),
    ("makes sense", "that is understandable"),
    ("what the hell", "what"),
    ("what on earth", "what"),
    ("out of nowhere", "suddenly"),
    ("hang in there", "endure it"),
    ("back off", "move away"),
    ("cut it out", "stop that"),
    ("pull yourself together", "regain your composure"),
    ("buy time", "delay"),
    ("sooner or later", "eventually"),
    ("over my dead body", "absolutely never"),
]

_REPLACEMENTS = _FIGURATIVE + _REPLACEMENTS

_PATTERNS = [
    (re.compile(r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b", re.IGNORECASE), plain)
    for phrase, plain in _REPLACEMENTS
]

_STANDALONE_PATTERNS = [
    (re.compile(r"^\W*" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\W*$", re.IGNORECASE), plain)
    for phrase, plain in _STANDALONE
]


def normalize(text: str, source_lang: str | None) -> str:
    if (source_lang or "").split("-")[0].lower() != "en":
        return text
    for pattern, plain in _STANDALONE_PATTERNS:
        if pattern.match(text):
            # A whole-utterance match replaces the segment outright; running
            # the inline passes afterwards could only re-edit our own output.
            return plain
    for pattern, plain in _PATTERNS:
        text = pattern.sub(plain, text)
    return text
