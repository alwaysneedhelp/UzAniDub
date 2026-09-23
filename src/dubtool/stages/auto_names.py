"""Automatically detects likely character/place names from a transcript,
so the user doesn't have to type out a whole show's cast list by hand.

Uses spaCy's general-purpose English NER (MIT-licensed, ~12MB model) — not
trained on anime specifically, so it works off ordinary cues (capitalization,
sentence position, syntax) rather than recognizing any name it's actually
seen before. That means it catches names that read like normal proper nouns
in context reasonably well (verified on real content: "Yahiko" in "the pain
that Yahiko suffered" was caught correctly), but can miss or only partially
catch invented compound names that happen to include ordinary English words
(e.g. "Almighty Push" was only half-caught, as "Push" — "Almighty" reads as
a plain adjective to a generic model). This is a real, known limitation:
treat auto-detection as a helpful default that cuts down manual typing for
the common case, not a substitute for a user-supplied cast list when
precision matters for a specific hard name.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_RELEVANT_LABELS = {"PERSON", "GPE", "ORG", "NORP", "FAC", "LOC"}
_MODEL_NAME = "en_core_web_sm"

_nlp = None  # lazy-loaded singleton — spaCy's model load isn't free


def detect_names(text: str) -> list[str]:
    """Returns likely proper nouns found in `text`, deduplicated, in the
    order first seen. Empty list (not an error) if spaCy/the model isn't
    installed or the text is empty — auto-detection is a nice-to-have, not
    something that should break the pipeline if unavailable."""
    global _nlp
    if not text.strip():
        return []

    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load(_MODEL_NAME)
        except Exception as e:
            log.warning(
                "automatic name detection unavailable (%s: %s) — proceeding with only "
                "explicitly-supplied names, if any", type(e).__name__, e,
            )
            _nlp = False  # sentinel: don't retry loading on every call
    if _nlp is False:
        return []

    doc = _nlp(text)
    seen: dict[str, None] = {}
    for ent in doc.ents:
        if ent.label_ in _RELEVANT_LABELS:
            name = ent.text.strip()
            if name and name.lower() not in (k.lower() for k in seen):
                seen[name] = None
    return list(seen)
