"""How pipeline hands auto-detected names to the translator.

spaCy's NER tags capitalized show terms as names ("Limitless Technique"),
and protecting those as identity glossary entries left English mid-line in
a real Gemini dub: "Shuning uchun Limitless Technique haqida ko'p bilasiz".
"""
from __future__ import annotations

from dubtool.pipeline import _protect_for


class _Literal:
    """A classic NMT backend: no handles_names."""


class _Llm:
    handles_names = True


GLOSSARY = {
    "Cursed Energy": "La'nat energiyasi",  # user-supplied term
    "Gojo": "Gojo",                        # user-supplied name
    "Limitless Technique": "Limitless Technique",  # auto-detected
    "Toji": "Toji",                                # auto-detected
}
AUTO = ["Limitless Technique", "Toji"]


def test_a_literal_backend_still_gets_every_entry_pinned():
    protected, term_map, hints = _protect_for(
        _Literal(), "Toji knows the Limitless Technique.", GLOSSARY, AUTO,
    )
    assert "Limitless" not in protected
    assert "Toji" not in protected
    assert sorted(term_map.values()) == ["Limitless Technique", "Toji"]
    assert hints == {}


def test_an_llm_backend_gets_auto_detected_names_as_hints_instead():
    protected, term_map, hints = _protect_for(
        _Llm(), "Toji knows the Limitless Technique.", GLOSSARY, AUTO,
    )
    assert protected == "Toji knows the Limitless Technique."
    assert term_map == {}
    assert hints == {"names": ["Limitless Technique", "Toji"]}


def test_user_glossary_entries_stay_enforced_for_an_llm_backend():
    protected, term_map, hints = _protect_for(
        _Llm(), "Gojo's Cursed Energy is huge.", GLOSSARY, AUTO,
    )
    assert "Gojo" not in protected
    assert "Cursed Energy" not in protected
    assert sorted(term_map.values()) == ["Gojo", "La'nat energiyasi"]
    assert hints == {"names": []}


def test_only_names_in_the_line_are_hinted():
    _, _, hints = _protect_for(_Llm(), "Toji is here.", GLOSSARY, AUTO)
    assert hints == {"names": ["Toji"]}
