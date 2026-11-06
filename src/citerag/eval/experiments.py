"""Experiment files: one YAML per run in configs/experiments/ (ARCHITECTURE section 13)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from citerag.config import Settings
from citerag.errors import ValidationFailedError

_SECRET_KEYS = {"openrouter_api_key", "gemini_api_key", "citerag_api_key"}


class Experiment(BaseModel):
    name: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    index_dir: Path | None = None
    retrieval: Literal["hybrid", "vector", "bm25"] = "hybrid"
    rerank: bool = False


def load_experiment(path: Path) -> Experiment:
    if not path.exists():
        raise ValidationFailedError(f"{path} not found")
    try:
        return Experiment.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    except ValidationError as exc:
        raise ValidationFailedError(f"{path}: {exc}") from exc


def apply_overrides(settings: Settings, overrides: dict[str, Any]) -> Settings:
    """model_copy skips validation, so each value is checked against its field here."""
    checked = {}
    for name, value in overrides.items():
        field = Settings.model_fields.get(name)
        if field is None or name in _SECRET_KEYS:
            raise ValidationFailedError(f"{name} cannot be set from an experiment file")
        try:
            checked[name] = TypeAdapter(field.annotation).validate_python(value)
        except ValidationError as exc:
            raise ValidationFailedError(f"{name}: {exc}") from exc
    return settings.model_copy(update=checked)
