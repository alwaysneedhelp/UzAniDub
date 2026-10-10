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


# --- whole-utterance tier -------------------------------------------------
# These phrases only have one dominant meaning as a complete line, so they
# can't be inline substitutions without breaking honest uses of the same
# words mid-sentence.


def test_standalone_affirmation_is_not_inverted():
    # Observed on real output: MADLAD rendered "Exactly!" as "Yo'q!" --
    # *"No!"* -- turning agreement into refusal. A fluent, confident
    # reversal of meaning is the one failure nothing downstream can catch.
    assert normalize("Exactly!", "en") == "yes, that is correct"


def test_standalone_match_ignores_surrounding_punctuation():
    for text in ("Exactly", "Exactly!", "...exactly?", "  Exactly.  "):
        assert normalize(text, "en") == "yes, that is correct"


def test_inline_use_of_a_standalone_phrase_is_left_alone():
    # The reason the tier exists: "exactly" mid-sentence means precisely,
    # and rewriting it to an affirmation would be a new bug.
    assert normalize("exactly five minutes remain", "en") == "exactly five minutes remain"


def test_standalone_tier_does_not_reprocess_its_own_output():
    # "got it" -> "I understand" must not then be re-matched by another rule.
    assert normalize("Got it.", "en") == "I understand"


# --- figurative tier ------------------------------------------------------


def test_figurative_death_phrase_keeps_the_speaker_as_subject():
    # Observed: became "I brought it to the edge of death" -- the figurative
    # "verge" scrambled both subject and meaning.
    assert normalize("But I finally grasped it on the verge of death.", "en") == (
        "But I finally grasped it when I was almost dead."
    )


def test_idiom_is_replaced_by_its_meaning_not_a_literal_gloss():
    # Deliberately avoids "sounds": the first version of this replacement
    # used it, and MADLAD read it as the noun and returned "more than his
    # voice" -- a replacement has to survive a literal reading too.
    assert normalize("Easier said than done!", "en") == (
        "that is very difficult to actually do!"
    )


def test_person_subject_high_is_disambiguated_to_the_drug_sense():
    # Observed: "Is he high?" translated as "Is he tall?".
    assert normalize("Is he high?", "en") == "is he on drugs?"


def test_bare_high_is_too_ambiguous_to_touch():
    # Only person-subject forms are rewritten; "high" alone legitimately
    # means elevated, loud, tall, expensive...
    for text in ("The wall is high", "a high price", "high above the city"):
        assert normalize(text, "en") == text


def test_figurative_tier_still_respects_non_english_source():
    text = "Easier said than done!"
    assert normalize(text, "ja") == text
