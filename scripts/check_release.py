"""Leakage and schema checks before a commit or a release. No network, no LLM, no model download.

    uv run python scripts/check_release.py

Fails (exit 1) when a tracked file is local or generated (.env, data, indexes, caches, eval results, model weights, local
tool config), when tracked text holds something that looks like an API key, when a commit message or tracked file carries an
assistant attribution line, or when the golden, cache-pair, experiment or .env.example files do not match the code.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATHS = [
    re.compile(p)
    for p in (
        r"(^|/)\.env$",
        r"(^|/)\.env\.(?!example$)[^/]+$",
        r"\.sqlite$",
        r"^data/",
        r"^eval/results/",
        r"(^|/)\.claude/",
        r"(^|/)\.mcp\.json$",
        r"\.(safetensors|onnx|pt|pth|ckpt)$",
        r"(^|/)chroma/",
    )
]
KEY_PATTERNS = [
    re.compile(p)
    for p in (
        r"sk-or-v1-[A-Za-z0-9]{20,}",
        r"AIza[0-9A-Za-z_\-]{30,}",
        r"\bsk-[A-Za-z0-9]{32,}",
        r"ghp_[A-Za-z0-9]{30,}",
    )
]
# a real trailer or footer, not a sentence that names the rule (CLAUDE.md does)
ATTRIBUTION = re.compile(
    r"^\s*Co-Authored-By:\s*\S|Generated with \[Claude Code\]\(|claude\.ai/code/session_",
    re.IGNORECASE | re.MULTILINE,
)
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".txt",
    ".example",
    ".cfg",
    ".ini",
    "",
}
SELF = "scripts/check_release.py"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def tracked() -> list[str]:
    return [line for line in git("ls-files").splitlines() if line]


def path_problems(files: list[str]) -> list[str]:
    return [
        f"tracked local or generated file: {f}" for f in files if any(p.search(f) for p in FORBIDDEN_PATHS)
    ]


def text_problems(files: list[str]) -> list[str]:
    found = []
    for name in files:
        path = ROOT / name
        if name == SELF or path.suffix not in TEXT_SUFFIXES or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(p.search(text) for p in KEY_PATTERNS):
            found.append(f"something that looks like an API key in {name}")  # never print the match itself
        if ATTRIBUTION.search(text):
            found.append(f"an attribution line in {name}")
    return found


def commit_problems() -> list[str]:
    bad = [line for line in git("log", "--format=%h %B").splitlines() if ATTRIBUTION.search(line)]
    return [f"an attribution line in a commit message: {line[:60]}" for line in bad]


def schema_problems() -> list[str]:
    sys.path.insert(0, str(ROOT / "src"))
    from citerag.config import Settings
    from citerag.eval.cache_tools import load_pairs
    from citerag.eval.experiments import load_experiment
    from citerag.eval.golden import load_golden

    found = []
    candidates = load_golden(ROOT / "eval/golden/candidates.jsonl")
    if len({i.id for i in candidates}) != len(candidates):
        found.append("duplicate ids in eval/golden/candidates.jsonl")
    if any(i.verified for i in candidates):
        found.append("a candidate is marked verified: only eval/golden/golden.jsonl may hold verified items")
    golden = ROOT / "eval/golden/golden.jsonl"
    if golden.exists() and not all(i.verified for i in load_golden(golden)):
        found.append("eval/golden/golden.jsonl holds an unverified item")
    if any(p.labelled or p.label for p in load_pairs(ROOT / "eval/cache_pairs.candidates.jsonl")):
        found.append("a cache-pair candidate carries a label: labels belong in eval/cache_pairs.jsonl")
    labelled = ROOT / "eval/cache_pairs.jsonl"
    if labelled.exists() and not all(p.labelled for p in load_pairs(labelled)):
        found.append("eval/cache_pairs.jsonl holds an unlabelled pair")
    for config in sorted((ROOT / "configs/experiments").glob("*.yaml")):
        load_experiment(config)
    keys = {
        line.split("=", 1)[0].strip()
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }
    fields = {name.upper() for name in Settings.model_fields}
    found += [f".env.example has {k}, which Settings does not read" for k in sorted(keys - fields)]
    found += [f"Settings reads {k}, missing from .env.example" for k in sorted(fields - keys)]
    return found


def main() -> int:
    files = tracked()
    problems = path_problems(files) + text_problems(files) + commit_problems() + schema_problems()
    for problem in problems:
        print(f"FAIL {problem}")
    print("release checks: " + ("FAILED" if problems else f"passed ({len(files)} tracked files)"))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
