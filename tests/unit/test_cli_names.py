from dubtool.cli import parse_names


def test_inline_comma_separated_list():
    assert parse_names("Naruto, Sasuke ,Sakura") == ["Naruto", "Sasuke", "Sakura"]


def test_single_inline_name():
    assert parse_names("Naruto") == ["Naruto"]


def test_reads_names_from_a_file_one_per_line(tmp_path):
    cast_file = tmp_path / "characters.txt"
    cast_file.write_text("Naruto\nSasuke\nSakura\n")

    assert parse_names(str(cast_file)) == ["Naruto", "Sasuke", "Sakura"]


def test_file_ignores_blank_lines_and_comments(tmp_path):
    cast_file = tmp_path / "characters.txt"
    cast_file.write_text(
        "# Main cast\n"
        "Naruto\n"
        "\n"
        "Sasuke\n"
        "# side characters below\n"
        "Yahiko\n"
    )

    assert parse_names(str(cast_file)) == ["Naruto", "Sasuke", "Yahiko"]
