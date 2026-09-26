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
