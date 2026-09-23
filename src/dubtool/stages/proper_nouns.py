"""Protects known proper nouns (character names, places, ...) from being
mangled by translation.

A general-purpose MT model has never seen most anime character names (or
any other names specific to the content being dubbed) and will sometimes
"translate" them into the nearest word it does know, or garble them
entirely. The fix used here is a standard localization technique: swap each
known name for a placeholder token before translation (one the MT model has
no reason to alter), then swap the placeholders back for the original names
afterward. This only protects names supplied up front via
`config.proper_nouns` — it can't discover names it wasn't told about.
"""
from __future__ import annotations

import re


def protect(text: str, proper_nouns: list[str]) -> tuple[str, dict[str, str]]:
    """Returns (text with known names replaced by placeholders, {placeholder: original_name})."""
    mapping: dict[str, str] = {}
    protected = text
    for i, name in enumerate(proper_nouns):
        if not name.strip():
            continue
        placeholder = f"XPRESERVEDNAME{i}X"
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        if pattern.search(protected):
            protected = pattern.sub(placeholder, protected)
            mapping[placeholder] = name
    return protected, mapping


def restore(text: str, mapping: dict[str, str]) -> str:
    """Swaps placeholders back for their original names after translation."""
    for placeholder, name in mapping.items():
        text = text.replace(placeholder, name)
    return text
