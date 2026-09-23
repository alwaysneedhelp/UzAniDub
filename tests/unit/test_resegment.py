from dubtool.stages.resegment import resegment_to_sentences
from dubtool.types import Segment, Word


def _w(text, start, end):
    return Word(text=text, start=start, end=end)


def test_merges_a_sentence_split_across_two_whisper_segments():
    # exactly the real-world case this was written for: a sentence cut
    # mid-way by whisper's own (non-sentence-aware) segment boundaries.
    segments = [
        Segment(
            start=0.0, end=1.0, speaker_id="SPEAKER_00", text="you for joining us.",
            words=[_w("you", 0.0, 0.2), _w("for", 0.2, 0.4), _w("joining", 0.4, 0.7), _w("us.", 0.7, 1.0)],
        ),
        Segment(
            start=1.0, end=2.5, speaker_id="SPEAKER_00", text="What my hope is",
            words=[_w("What", 1.0, 1.3), _w("my", 1.3, 1.5), _w("hope", 1.5, 1.8), _w("is.", 1.8, 2.5)],
        ),
    ]

    result = resegment_to_sentences(segments)

    assert len(result) == 2
    assert result[0].text == "you for joining us."
    assert result[1].text == "What my hope is."
    assert result[0].start == 0.0 and result[0].end == 1.0
    assert result[1].start == 1.0 and result[1].end == 2.5


def test_splits_one_segment_containing_two_sentences():
    segments = [
        Segment(
            start=0.0, end=2.0, speaker_id="SPEAKER_00", text="Hi there. How are you?",
            words=[_w("Hi", 0.0, 0.2), _w("there.", 0.2, 0.5), _w("How", 1.0, 1.2), _w("are", 1.2, 1.4), _w("you?", 1.4, 2.0)],
        ),
    ]

    result = resegment_to_sentences(segments)

    assert [s.text for s in result] == ["Hi there.", "How are you?"]
    assert result[1].start == 1.0  # preserves the real gap between sentences


def test_never_merges_across_a_speaker_change():
    segments = [
        Segment(
            start=0.0, end=1.0, speaker_id="SPEAKER_00", text="hello",
            words=[_w("hello", 0.0, 1.0)],  # no terminal punctuation
        ),
        Segment(
            start=1.0, end=2.0, speaker_id="SPEAKER_01", text="hi",
            words=[_w("hi", 1.0, 2.0)],
        ),
    ]

    result = resegment_to_sentences(segments)

    assert len(result) == 2
    assert result[0].speaker_id == "SPEAKER_00"
    assert result[1].speaker_id == "SPEAKER_01"


def test_falls_back_unchanged_when_words_are_missing():
    segments = [Segment(start=0.0, end=1.0, speaker_id="SPEAKER_00", text="no word timestamps")]
    assert resegment_to_sentences(segments) == segments


def test_empty_input():
    assert resegment_to_sentences([]) == []


def test_forces_a_split_on_long_punctuation_free_runs():
    # the real regression this guards against: Whisper produced 100+ seconds
    # of transcript with zero terminal punctuation for a fast, conversational
    # speaking style, and an earlier uncapped version of this function merged
    # all of it into one enormous, effectively untranslatable segment.
    words = []
    t = 0.0
    for i in range(60):  # 60 words, no punctuation anywhere, spans well over max_duration
        words.append(_w(f"word{i}", t, t + 0.3))
        t += 0.4  # small, uniform gaps — nothing that looks like a real pause
    segments = [Segment(start=0.0, end=t, speaker_id="SPEAKER_00", text="...", words=words)]

    result = resegment_to_sentences(segments, max_duration=5.0)

    assert len(result) > 1
    for seg in result:
        assert seg.duration <= 5.0 + 0.4  # allow one word's worth of slack at the boundary
    # every original word must show up exactly once across the split chunks
    rejoined = " ".join(seg.text for seg in result)
    assert rejoined == " ".join(w.text for w in words)


def test_does_not_explode_into_one_word_per_segment_on_zero_gap_speech():
    # the exact real regression: continuous, fast speech has ~0s gaps
    # between every word, so "pick the largest gap" with no minimum
    # threshold degenerates into picking the same trivial index every time
    # and peeling off one word per split — a ~20-word run turning into ~20
    # single-word segments, each translated with no context.
    words = []
    t = 0.0
    for i in range(50):
        words.append(_w(f"word{i}", t, t + 0.3))
        t += 0.3  # zero gap: next word starts exactly when the last ends
    segments = [Segment(start=0.0, end=t, speaker_id="SPEAKER_00", text="...", words=words)]

    result = resegment_to_sentences(segments, max_duration=5.0)

    # ~15s of audio at a 5s cap should be ~3 chunks, not 50
    assert len(result) <= 5
    assert all(len(seg.words) > 1 for seg in result), "must not degenerate to single-word segments"


def test_forced_split_prefers_the_largest_pause():
    words = [
        _w("alpha", 0.0, 1.0),
        _w("beta", 1.1, 2.0),   # small gap (0.1s) after this one
        _w("gamma", 4.0, 5.0),  # big gap (2.0s) before this one — should split here
        _w("delta", 5.1, 6.0),
    ]
    segments = [Segment(start=0.0, end=6.0, speaker_id="SPEAKER_00", text="...", words=words)]

    result = resegment_to_sentences(segments, max_duration=3.0)

    assert result[0].text == "alpha beta"
    assert result[1].text == "gamma delta"
