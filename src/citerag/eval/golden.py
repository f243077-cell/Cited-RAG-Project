"""Load golden items. Only scripts/accept_golden.py sets `verified`; `reviewed_by` records who checked the item."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from citerag.errors import ValidationFailedError
from citerag.schemas import GoldenItem


def load_golden(path: Path, require_verified: bool = False) -> list[GoldenItem]:
    if not path.exists():
        raise ValidationFailedError(f"{path} not found")
    items: list[GoldenItem] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(GoldenItem.model_validate_json(line))
        except ValidationError as exc:
            raise ValidationFailedError(f"{path} line {number}: {exc}") from exc
    ids = [item.id for item in items]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValidationFailedError(f"{path}: duplicate ids {duplicates}")
    if require_verified:
        unverified = [item.id for item in items if not item.verified]
        if unverified:
            raise ValidationFailedError(f"{path}: not verified by a human: {unverified}")
    return items
