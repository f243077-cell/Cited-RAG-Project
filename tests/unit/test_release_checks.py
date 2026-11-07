"""scripts/check_release.py: the leakage and schema checks every commit must pass."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_release", ROOT / "scripts" / "check_release.py")
check = importlib.util.module_from_spec(spec)
sys.modules["check_release"] = check
spec.loader.exec_module(check)


@pytest.mark.parametrize(
    "name",
    [".env", "sub/.env", ".env.local", "data/index/x", "data/logs/app.sqlite", "eval/results/run/items.jsonl",
     ".claude/settings.local.json", ".mcp.json", "models/bge.safetensors", "x/chroma/y"],
)  # fmt: skip
def test_local_and_generated_paths_are_caught(name):
    assert check.path_problems([name])


@pytest.mark.parametrize(
    "name", [".env.example", "src/citerag/config.py", "eval/golden/candidates.jsonl", "README.md"]
)
def test_project_paths_pass(name):
    assert check.path_problems([name]) == []


def test_a_key_pattern_or_an_attribution_line_is_caught_without_echoing_it(tmp_path, monkeypatch):
    monkeypatch.setattr(check, "ROOT", tmp_path)
    (tmp_path / "a.md").write_text("key: sk-or-v1-" + "a" * 40, encoding="utf-8")
    (tmp_path / "b.md").write_text("Co-Authored-By: someone", encoding="utf-8")
    problems = check.text_problems(["a.md", "b.md"])
    assert problems == ["something that looks like an API key in a.md", "an attribution line in b.md"]
    assert all("aaaa" not in p for p in problems)


def test_the_repository_passes_every_check():
    assert check.path_problems(check.tracked()) == []
    assert check.text_problems(check.tracked()) == []
    assert check.commit_problems() == []
    assert check.schema_problems() == []
