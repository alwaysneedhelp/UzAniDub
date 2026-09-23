"""Smoke test only — actual translation needs the ~3GB CTranslate2 model
downloaded and loaded, which doesn't belong in the fast unit suite. This
just checks the class is importable, constructs without downloading
anything (loading is lazy), and satisfies the Translator protocol shape.
Real translation quality is exercised by the integration run instead.
"""
from dubtool.backends.translate_madlad import MadladTranslator


def test_constructs_without_downloading_anything():
    translator = MadladTranslator()
    assert translator._translator is None
    assert translator._tokenizer is None


def test_empty_text_short_circuits_without_loading_the_model():
    translator = MadladTranslator()
    assert translator.translate("", source_lang="en", target_lang="uz") == ""
    # still not loaded — an empty string must never trigger a model load
    assert translator._translator is None
