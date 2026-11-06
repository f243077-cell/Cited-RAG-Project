"""Prompts live in *.md files in this folder. The first line of each is `version: <n>`, logged with every request."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent
_HEADER = re.compile(r"\Aversion:\s*(\S+)\s*\n")


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    text: str


def parse_prompt(name: str, raw: str) -> Prompt:
    match = _HEADER.match(raw)
    if match is None:
        raise ValueError(f"prompt {name!r} has no `version:` header on its first line")
    return Prompt(name=name, version=match.group(1), text=raw[match.end() :].strip())


def load_prompt(name: str) -> Prompt:
    return parse_prompt(name, (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8"))
