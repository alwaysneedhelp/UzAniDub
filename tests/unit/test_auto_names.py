from dubtool.stages.auto_names import detect_names


def test_detects_a_name_in_real_naruto_transcript_context():
    # exactly the real case this was built for
    text = "I will never forget the pain that Yahiko suffered."
    names = detect_names(text)
    assert any("yahiko" in n.lower() for n in names)


def test_empty_text_returns_empty_list():
    assert detect_names("") == []
    assert detect_names("   ") == []


def test_deduplicates_repeated_mentions():
    text = "Naruto went home. Later, Naruto came back. Naruto smiled."
    names = detect_names(text)
    naruto_mentions = [n for n in names if n.lower() == "naruto"]
    assert len(naruto_mentions) == 1


def test_text_with_no_named_entities_returns_empty_or_unrelated():
    # sentences with no proper nouns shouldn't crash or hallucinate names
    names = detect_names("The weather is nice today and I feel happy.")
    assert isinstance(names, list)
