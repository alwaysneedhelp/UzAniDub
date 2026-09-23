from dubtool.stages.proper_nouns import protect, restore


def test_protects_and_restores_a_single_name():
    text = "Naruto went to see Sasuke at the ramen shop."
    protected, mapping = protect(text, ["Naruto", "Sasuke"])

    assert "Naruto" not in protected
    assert "Sasuke" not in protected
    assert restore(protected, mapping) == text


def test_case_insensitive_matching():
    text = "naruto and NARUTO and Naruto all refer to the same person."
    protected, mapping = protect(text, ["Naruto"])

    assert "naruto" not in protected.lower()
    restored = restore(protected, mapping)
    # every occurrence gets normalized to the canonical spelling we were given
    assert restored.count("Naruto") == 3


def test_names_not_present_are_ignored():
    text = "Hello world."
    protected, mapping = protect(text, ["Naruto", "Sasuke"])

    assert protected == text
    assert mapping == {}


def test_empty_names_list():
    text = "Hello world."
    protected, mapping = protect(text, [])

    assert protected == text
    assert mapping == {}


def test_restore_with_empty_mapping_is_a_no_op():
    assert restore("some text", {}) == "some text"
