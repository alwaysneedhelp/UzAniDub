from dubtool.stages.glossary import protect, restore


def test_name_entries_pass_through_unchanged():
    # a character name is just an entry mapping to itself
    text = "Naruto went to see Sasuke at the ramen shop."
    protected, mapping = protect(text, {"Naruto": "Naruto", "Sasuke": "Sasuke"})

    assert "Naruto" not in protected
    assert "Sasuke" not in protected
    assert restore(protected, mapping) == text


def test_term_entries_get_a_different_fixed_translation():
    text = "He gathered his Cursed Energy before using the Cursed Technique."
    glossary = {
        "Cursed Energy": "La'nat energiyasi",
        "Cursed Technique": "La'nat texnikasi",
    }
    protected, mapping = protect(text, glossary)
    restored = restore(protected, mapping)

    assert "La'nat energiyasi" in restored
    assert "La'nat texnikasi" in restored
    assert "Cursed Energy" not in restored
    assert "Cursed Technique" not in restored


def test_consistent_across_repeated_occurrences():
    # the actual problem this solves: the same term shouldn't come out
    # differently in different places just because a generic MT model has
    # no memory across calls -- here we don't even reach a real translator,
    # but confirm every occurrence maps to the exact same fixed string.
    text = "Cursed Energy flows. More Cursed Energy appears. Even more Cursed Energy."
    protected, mapping = protect(text, {"Cursed Energy": "La'nat energiyasi"})
    restored = restore(protected, mapping)

    assert restored.count("La'nat energiyasi") == 3


def test_case_insensitive_matching():
    text = "cursed energy, CURSED ENERGY, and Cursed Energy all refer to the same thing."
    protected, mapping = protect(text, {"Cursed Energy": "La'nat energiyasi"})
    restored = restore(protected, mapping)

    assert restored.count("La'nat energiyasi") == 3


def test_terms_not_present_are_ignored():
    text = "Hello world."
    protected, mapping = protect(text, {"Naruto": "Naruto", "Cursed Energy": "La'nat energiyasi"})

    assert protected == text
    assert mapping == {}


def test_empty_glossary():
    text = "Hello world."
    protected, mapping = protect(text, {})

    assert protected == text
    assert mapping == {}


def test_restore_with_empty_mapping_is_a_no_op():
    assert restore("some text", {}) == "some text"


def test_restores_a_placeholder_whose_letters_came_back_altered():
    # An instruction-following backend treats the sentinel as text to rewrite
    # and returns it a letter short. Observed on real Gemini output as
    # "XPRSERVEDTERM40X", which an exact replace left in the finished line
    # for the TTS to read aloud. Only the digits are dependable.
    protected, mapping = protect(
        "He used the Reverse Cursed Technique.",
        {"Reverse Cursed Technique": "Teskari la'nat texnikasi"},
    )
    mangled = protected.replace("XPRESERVEDTERM", "XPRSERVEDTERM")
    assert mangled != protected          # guard: the test is exercising the bug
    assert "Teskari la'nat texnikasi" in restore(mangled, mapping)
    assert "SERVEDTERM" not in restore(mangled, mapping)


def test_leaves_an_unknown_placeholder_like_token_alone():
    # Don't rewrite something that merely looks like a sentinel but has no
    # entry -- the index has to actually be one we issued.
    text = "Model X100X is unrelated."
    _, mapping = protect("nothing here", {"Gojo": "Gojo"})
    assert restore(text, mapping) == text


def test_a_longer_term_is_not_consumed_by_a_shorter_one_inside_it():
    # File order put "Cursed Technique" before "Reverse Cursed Technique";
    # the shorter one matched first, the longer entry never applied, and the
    # line came out as "returning to Cursed Technique".
    glossary = {
        "Cursed Technique": "La'nat texnikasi",
        "Reverse Cursed Technique": "Teskari la'nat texnikasi",
    }
    protected, mapping = protect("Reverse Cursed Technique.", glossary)
    assert restore(protected, mapping) == "Teskari la'nat texnikasi."


def test_a_term_is_not_matched_inside_a_longer_word():
    # "Curse" matched inside "Cursed" and left "La'natd" in the line.
    protected, mapping = protect("you're cursed", {"Curse": "La'nat"})
    assert restore(protected, mapping) == "you're cursed"


def test_an_english_plural_is_absorbed_into_the_term():
    # Used to come out as "La'nat ruhis".
    protected, mapping = protect("two Cursed Spirits", {"Cursed Spirit": "La'nat ruhi"})
    assert restore(protected, mapping) == "two La'nat ruhi"


def test_a_possessive_still_protects_the_name():
    protected, _ = protect("Gojo's eyes", {"Gojo": "Gojo"})
    assert "Gojo" not in protected
