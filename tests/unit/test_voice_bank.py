import json
import numpy as np
import soundfile as sf

from dubtool.stages import voice_bank
from dubtool.stages.voice_bank import VoiceBank


def _write_wav(path, sr=16000):
    sf.write(str(path), np.zeros(sr, dtype=np.float32), sr)
    return path


def _fake_embed_factory(mapping):
    """Returns a fake _embed(path) -> np.ndarray using a path->vector dict,
    so tests control similarity deterministically instead of depending on
    Resemblyzer's actual behavior on synthetic (non-speech) test audio."""
    def _fake_embed(path):
        return np.array(mapping[str(path)], dtype=np.float32)
    return _fake_embed


def test_first_clip_registers_a_new_voice(tmp_path, monkeypatch):
    clip_path = _write_wav(tmp_path / "clip.wav")
    monkeypatch.setattr(voice_bank, "_embed", _fake_embed_factory({str(clip_path): [1.0, 0.0, 0.0]}))

    bank = VoiceBank(tmp_path / "bank")
    result = bank.find_or_add(clip_path, "hello there", candidate_score=1.0)

    assert result.path.name == "speaker_001.wav"
    assert result.text == "hello there"
    assert (tmp_path / "bank" / "speaker_001.json").exists()


def test_similar_voice_matches_existing_entry(tmp_path, monkeypatch):
    clip_a = _write_wav(tmp_path / "a.wav")
    clip_b = _write_wav(tmp_path / "b.wav")
    # nearly identical vectors -> high cosine similarity
    monkeypatch.setattr(voice_bank, "_embed", _fake_embed_factory({
        str(clip_a): [1.0, 0.0, 0.0],
        str(clip_b): [0.99, 0.01, 0.0],
    }))

    bank = VoiceBank(tmp_path / "bank", similarity_threshold=0.9)
    first = bank.find_or_add(clip_a, "first text", candidate_score=0.5)
    second = bank.find_or_add(clip_b, "second text", candidate_score=0.3)

    # second clip matched the same voice, so it reuses the first entry's id
    assert first.path == second.path
    # lower score than the stored one -> text NOT replaced
    assert second.text == "first text"


def test_dissimilar_voice_registers_as_a_new_entry(tmp_path, monkeypatch):
    clip_a = _write_wav(tmp_path / "a.wav")
    clip_b = _write_wav(tmp_path / "b.wav")
    # orthogonal vectors -> zero cosine similarity
    monkeypatch.setattr(voice_bank, "_embed", _fake_embed_factory({
        str(clip_a): [1.0, 0.0, 0.0],
        str(clip_b): [0.0, 1.0, 0.0],
    }))

    bank = VoiceBank(tmp_path / "bank", similarity_threshold=0.75)
    first = bank.find_or_add(clip_a, "voice A", candidate_score=0.5)
    second = bank.find_or_add(clip_b, "voice B", candidate_score=0.5)

    assert first.path != second.path
    assert {first.path.name, second.path.name} == {"speaker_001.wav", "speaker_002.wav"}


def test_matched_entry_updates_when_new_clip_scores_higher(tmp_path, monkeypatch):
    clip_a = _write_wav(tmp_path / "a.wav")
    clip_b = _write_wav(tmp_path / "b.wav")
    monkeypatch.setattr(voice_bank, "_embed", _fake_embed_factory({
        str(clip_a): [1.0, 0.0, 0.0],
        str(clip_b): [1.0, 0.0, 0.0],
    }))

    bank = VoiceBank(tmp_path / "bank", similarity_threshold=0.9)
    bank.find_or_add(clip_a, "old text", candidate_score=0.2)
    result = bank.find_or_add(clip_b, "better text", candidate_score=0.9)

    assert result.text == "better text"


def test_a_voice_claimed_this_run_cannot_be_matched_or_overwritten_by_another_character(tmp_path, monkeypatch):
    # On a real clip a side character scored >=0.75 against the main
    # character's fresh entry and, scoring higher, replaced its audio: both
    # then cloned the side character's 2.5s clip.
    main = _write_wav(tmp_path / "main.wav")
    side = _write_wav(tmp_path / "side.wav")
    monkeypatch.setattr(voice_bank, "_embed", _fake_embed_factory({
        str(main): [1.0, 0.0, 0.0],
        str(side): [0.99, 0.01, 0.0],      # similar enough to match
    }))
    bank = VoiceBank(tmp_path / "bank", similarity_threshold=0.9)

    first = bank.find_or_add(main, "main", candidate_score=0.1)
    claimed = {first.path.stem}
    second = bank.find_or_add(side, "side", candidate_score=0.9, exclude=claimed)

    assert second.path != first.path
    assert json.loads((tmp_path / "bank" / f"{first.path.stem}.json").read_text())["text"] == "main"


def test_a_voice_from_an_earlier_run_still_matches_when_nothing_has_claimed_it(tmp_path, monkeypatch):
    # The exclusion is per run; cross-episode matching is the bank's purpose.
    ep1 = _write_wav(tmp_path / "ep1.wav")
    ep2 = _write_wav(tmp_path / "ep2.wav")
    monkeypatch.setattr(voice_bank, "_embed", _fake_embed_factory({
        str(ep1): [1.0, 0.0, 0.0],
        str(ep2): [0.99, 0.01, 0.0],
    }))
    VoiceBank(tmp_path / "bank", similarity_threshold=0.9).find_or_add(ep1, "ep1", candidate_score=0.5)

    again = VoiceBank(tmp_path / "bank", similarity_threshold=0.9).find_or_add(ep2, "ep2", candidate_score=0.1, exclude=set())

    assert again.path.stem == "speaker_001"
