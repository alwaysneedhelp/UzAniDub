from dubtool.stages.idioms import normalize


def test_rewrites_known_slang_for_english_source():
    assert normalize("Wait, for real?", "en") == "Wait, really?"


def test_case_insensitive():
    assert normalize("FOR REAL though", "en") == "really though"


def test_longer_phrase_takes_priority_over_shorter_one():
    assert normalize("for real for real, I mean it", "en") == "truly, I mean it"


def test_leaves_non_english_source_untouched():
    text = "for real, this is definitely japanese text pretending"
    assert normalize(text, "ja") == text


def test_leaves_unmatched_text_untouched():
    text = "Nothing slangy here."
    assert normalize(text, "en") == text


def test_handles_source_lang_with_region_suffix():
    assert normalize("for real", "en-US") == "really"


def test_handles_missing_source_lang():
    text = "for real"
    assert normalize(text, None) == text
