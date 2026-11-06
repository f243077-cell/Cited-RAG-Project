"""Load configs/pricing.yaml and expose the reference entry as pricing["_reference"] (see estimate_cost_usd)."""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "configs" / "pricing.yaml"


def load_pricing(path: Path = DEFAULT_PATH) -> dict[str, dict[str, float]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    models = data.get("models") or {}
    reference = data.get("reference_model")
    if not data.get("retrieved_on") or not data.get("source_url"):
        raise ValueError(f"{path}: retrieved_on and source_url must be filled in")
    if reference not in models:
        raise ValueError(f"{path}: reference_model {reference!r} has no entry under models")
    return {**models, "_reference": models[reference]}
