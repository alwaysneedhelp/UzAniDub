from dubtool.cli import parse_glossary


def test_loads_term_translation_pairs(tmp_path):
    glossary_file = tmp_path / "glossary.yaml"
    glossary_file.write_text(
        "Gojo: Gojo\n"
        "Cursed Energy: La'nat energiyasi\n"
    )

    assert parse_glossary(glossary_file) == {
        "Gojo": "Gojo",
        "Cursed Energy": "La'nat energiyasi",
    }


def test_empty_file_yields_empty_glossary(tmp_path):
    glossary_file = tmp_path / "glossary.yaml"
    glossary_file.write_text("")

    assert parse_glossary(glossary_file) == {}


def test_non_mapping_yaml_raises(tmp_path):
    glossary_file = tmp_path / "glossary.yaml"
    glossary_file.write_text("- Gojo\n- Sukuna\n")

    try:
        parse_glossary(glossary_file)
        assert False, "expected ValueError"
    except ValueError:
        pass
