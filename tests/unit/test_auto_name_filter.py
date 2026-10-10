"""Auto-detected names must not shadow terms the glossary already has.

Detection returned "the Reverse Cursed Technique"; the article kept it from
matching the existing entry, so it was added as a new pass-through name,
and as the longer term it beat the real translation -- English words left
in the middle of an Uzbek line, read aloud by the TTS.
"""
from dubtool.pipeline import _names_not_in_glossary

GLOSSARY = {"Reverse Cursed Technique": "Teskari la'nat texnikasi", "Gojo": "Gojo"}


def test_a_name_overlapping_a_glossary_term_is_not_added():
    assert _names_not_in_glossary(["the Reverse Cursed Technique"], GLOSSARY) == []


def test_a_longer_form_of_a_known_name_is_not_added():
    assert _names_not_in_glossary(["Gojo Satoru"], GLOSSARY) == []


def test_a_genuinely_new_name_is_kept_without_its_article():
    assert _names_not_in_glossary(["the Shibuya Incident"], GLOSSARY) == ["Shibuya Incident"]


def test_duplicates_are_kept_once():
    assert _names_not_in_glossary(["Mahito", "mahito"], GLOSSARY) == ["Mahito"]
