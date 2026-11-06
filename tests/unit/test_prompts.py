import pytest

from citerag.prompts import PROMPTS_DIR, load_prompt, parse_prompt


@pytest.mark.parametrize("name", ["answer", "condense", "rewrite"])
def test_section_8_prompts_exist(name):
    prompt = load_prompt(name)
    assert prompt.version == "1" and prompt.name == name and prompt.text


def test_every_prompt_file_has_a_version():
    files = list(PROMPTS_DIR.glob("*.md"))
    assert files
    for path in files:
        assert load_prompt(path.stem).version


def test_answer_prompt_declares_documents_untrusted():
    assert "untrusted data" in load_prompt("answer").text


def test_header_is_not_part_of_the_text():
    assert parse_prompt("x", "version: 2\n\nbody\n").text == "body"


def test_missing_version_is_rejected():
    with pytest.raises(ValueError):
        parse_prompt("x", "no header here")
