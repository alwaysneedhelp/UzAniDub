"""Re-groups word-level ASR output into complete sentences before translation.

Whisper's own segment boundaries are driven by its internal chunking, not
sentence structure — they can (and do, in real audio) cut a sentence in
half, e.g. "...you for joining us. What my hope is and I'm very pleased to
be with all of you" ending up as its own segment starting mid-sentence.
Feeding a translator a fragment like that instead of two clean sentences
measurably hurts translation quality (dropped clauses, repetition) — this
was diagnosed from a real run where exactly the first two segments, cut
this way, produced the worst translations in the whole file while
everything downstream (where sentence boundaries happened to line up with
segment boundaries by luck) was fine.

This only regroups the *words*, it doesn't re-run ASR — it just trusts
word-level timestamps more than segment-level ones, which is reasonable
since faster-whisper's word timestamps come from the same forced-alignment
process either way.

`max_duration` matters in practice, not just in theory: a real run on a
fast, conversational speaking style produced over 100 seconds of Whisper
transcript with *zero* terminal punctuation, which an earlier, uncapped
version of this function merged into one enormous segment — effectively
untranslatable as one unit and impossible to time-align sanely against a
single TTS call. When forced to split without punctuation, this looks for
the largest inter-word pause to approximate a real clause boundary — but
only if that pause clears `min_gap`. A first version skipped that
threshold and shipped a much worse bug on the same real audio: continuous,
fast speech has inter-word gaps of ~0s throughout, so "largest gap" among a
bunch of uniformly-tiny gaps degenerates to picking whichever index happens
to compare largest first — in practice, index 1 — every single time,
peeling off one word per split and exploding a ~20-segment transcript into
~200 single-word ones (each translated with essentially no context, which
is also why isolated words occasionally came back as garbage like a raw
"%s" template artifact). Below `min_gap`, there is no real pause to anchor
on, so this falls back to cutting wherever the run first reaches
`max_duration`, keeping forced chunks close to full-sized instead.
"""
from __future__ import annotations

from dubtool.types import Segment, Word

_SENTENCE_ENDINGS = (".", "!", "?", "。", "！", "？", "؟")


def resegment_to_sentences(segments: list[Segment], max_duration: float = 12.0) -> list[Segment]:
    """Returns a new list of Segments, one per complete sentence (or, absent
    punctuation for too long, one per forced sub-sentence chunk under
    `max_duration`), built from the word-level timestamps of the input
    segments. Falls back to returning `segments` unchanged if any segment
    has no word-level data (e.g. a transcriber backend that doesn't
    populate `.words`)."""
    if not segments or any(not seg.words for seg in segments):
        return segments

    tagged_words = [
        (word, seg.speaker_id, seg.detected_language) for seg in segments for word in seg.words
    ]

    chunks: list[tuple[list[Word], str, str | None]] = []
    current_words: list[Word] = []
    current_speaker = tagged_words[0][1]
    current_lang = tagged_words[0][2]

    for word, speaker_id, detected_language in tagged_words:
        if speaker_id != current_speaker and current_words:
            chunks.append((current_words, current_speaker, current_lang))
            current_words = []
        current_speaker = speaker_id
        current_lang = detected_language
        current_words.append(word)

        if word.text.strip().endswith(_SENTENCE_ENDINGS):
            chunks.append((current_words, current_speaker, current_lang))
            current_words = []
        elif current_words[-1].end - current_words[0].start > max_duration:
            split_idx = _best_forced_split_index(current_words, max_duration)
            chunks.append((current_words[:split_idx], current_speaker, current_lang))
            current_words = current_words[split_idx:]

    if current_words:
        chunks.append((current_words, current_speaker, current_lang))

    result = []
    for words, speaker_id, detected_language in chunks:
        result.append(
            Segment(
                start=words[0].start,
                end=words[-1].end,
                speaker_id=speaker_id,
                text=" ".join(w.text for w in words).strip(),
                words=words,
                detected_language=detected_language,
            )
        )
    return result


def _best_forced_split_index(words: list[Word], max_duration: float, min_gap: float = 0.15) -> int:
    """Index i (always >= 1) such that splitting into words[:i], words[i:]
    is a reasonable place to force a boundary given `words` has exceeded
    `max_duration` with no sentence-ending punctuation.

    Prefers the largest inter-word pause, but only if it's a *real* pause
    (>= min_gap) — otherwise every gap in a continuous, fast-talking run is
    ~0s and "largest" becomes meaningless noise, which is exactly what
    produced the 1-word-per-segment explosion this function is written
    against (see module docstring). Without a qualifying gap, falls back to
    cutting at whichever word first reaches max_duration, so forced chunks
    stay close to full-sized instead of degenerating to single words.
    """
    start_time = words[0].start
    fallback_index = len(words) - 1
    for i, w in enumerate(words):
        if w.end - start_time >= max_duration:
            fallback_index = max(1, i)
            break

    best_index, best_gap = None, min_gap
    for i in range(1, len(words)):
        gap = words[i].start - words[i - 1].end
        if gap >= best_gap:
            best_gap, best_index = gap, i

    return best_index if best_index is not None else fallback_index
