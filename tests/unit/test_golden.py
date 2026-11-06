import json

import pytest

from citerag.errors import ValidationFailedError
from citerag.eval.golden import load_golden


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    return path


ITEM = {"id": "a-1", "question": "q", "type": "answerable", "gold_sources": [{"doc": "a.pdf", "page": 2}]}


def test_loads_items_and_skips_blank_lines(tmp_path):
    items = load_golden(_write(tmp_path / "g.jsonl", [ITEM, {**ITEM, "id": "a-2"}]))
    assert [i.id for i in items] == ["a-1", "a-2"] and items[0].verified is False
    assert items[0].gold_sources[0].page == 2


def test_require_verified_rejects_unverified_items(tmp_path):
    path = _write(tmp_path / "g.jsonl", [ITEM])
    with pytest.raises(ValidationFailedError, match="a-1"):
        load_golden(path, require_verified=True)
    good = _write(tmp_path / "ok.jsonl", [{**ITEM, "verified": True}])
    assert load_golden(good, require_verified=True)[0].verified is True


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(ValidationFailedError, match="duplicate"):
        load_golden(_write(tmp_path / "g.jsonl", [ITEM, ITEM]))


def test_bad_line_names_its_number(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps(ITEM) + '\n{"id": "x"}\n', encoding="utf-8")
    with pytest.raises(ValidationFailedError, match="line 2"):
        load_golden(path)


def test_missing_file_is_a_typed_error(tmp_path):
    with pytest.raises(ValidationFailedError, match="not found"):
        load_golden(tmp_path / "missing.jsonl")
