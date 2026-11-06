"""List OpenRouter's zero-priced models by capability and suggest one model per role (PRD F-OPS-7).

The public model list needs no API key. This script never writes .env: paste the printed lines yourself.
    uv run python scripts/pick_free_models.py
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import typer

MODELS_URL = "https://openrouter.ai/api/v1/models"
ROLES = ("fast", "strong", "vision", "judge")


class NoModelError(RuntimeError):
    """No free model satisfies a role's needs today."""


@dataclass(frozen=True)
class ModelInfo:
    id: str
    context_length: int
    image: bool
    structured_outputs: bool
    response_format: bool
    tools: bool
    temperature: bool
    mandatory_reasoning: bool
    expires: str | None


def parse_model(raw: dict) -> ModelInfo:
    params = set(raw.get("supported_parameters") or [])
    return ModelInfo(
        id=raw["id"],
        context_length=int(raw.get("context_length") or 0),
        image="image" in ((raw.get("architecture") or {}).get("input_modalities") or []),
        structured_outputs="structured_outputs" in params,
        response_format="response_format" in params,
        tools="tools" in params,
        temperature="temperature" in params,
        mandatory_reasoning=bool((raw.get("reasoning") or {}).get("mandatory")),
        expires=raw.get("expiration_date"),
    )


def _price_is_zero(value: object) -> bool:
    try:
        return float(value) == 0  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _is_free_chat_model(raw: dict) -> bool:
    pricing = raw.get("pricing") or {}
    outputs = (raw.get("architecture") or {}).get("output_modalities") or []
    return (
        _price_is_zero(pricing.get("prompt"))
        and _price_is_zero(pricing.get("completion"))
        and not raw["id"].startswith("openrouter/")  # routers pick a different model per request
        and outputs == ["text"]  # music and video generators are also listed at price 0
    )


def free_models(raw_models: list[dict]) -> list[ModelInfo]:
    return [parse_model(r) for r in raw_models if _is_free_chat_model(r)]


def _can_structure(m: ModelInfo) -> bool:
    return m.structured_outputs or m.response_format or m.tools


def _rank(m: ModelInfo) -> tuple:
    return (m.structured_outputs, m.response_format, m.tools, not m.mandatory_reasoning, m.context_length)


def pick_roles(models: list[ModelInfo]) -> dict[str, str]:
    """Only stable :free ids qualify: their limits are documented, and a model with a removal date breaks pinning."""
    pool = sorted(
        (m for m in models if m.id.endswith(":free") and not m.expires and _can_structure(m)),
        key=_rank,
        reverse=True,
    )
    if not pool:
        raise NoModelError("no stable :free model supports structured output or tools")
    vision = [m for m in pool if m.image]
    if not vision:
        raise NoModelError("no stable :free model accepts image input")
    strong = pool[0].id
    judge = next((m.id for m in pool if m.id != strong), strong)  # self-preference bias: differ when possible
    return {"fast": strong, "strong": strong, "vision": vision[0].id, "judge": judge}


def env_lines(picks: dict[str, str]) -> list[str]:
    return [f"MODEL_{role.upper()}={picks[role]}" for role in ROLES]


def _yes(value: bool) -> str:
    return "y" if value else "-"


def table(models: list[ModelInfo]) -> list[str]:
    head = f"{'model':<52} {'context':>9} img so rf tools temp think expires"
    rows = [head, "-" * len(head)]
    for m in sorted(models, key=_rank, reverse=True):
        rows.append(
            f"{m.id:<52} {m.context_length:>9} {_yes(m.image):>3} {_yes(m.structured_outputs):>2}"
            f" {_yes(m.response_format):>2} {_yes(m.tools):>5} {_yes(m.temperature):>4}"
            f" {_yes(m.mandatory_reasoning):>5} {m.expires or '-'}"
        )
    return rows


def main() -> None:
    response = httpx.get(MODELS_URL, timeout=30)
    response.raise_for_status()
    models = free_models(response.json()["data"])
    for row in table(models):
        typer.echo(row)
    typer.echo(
        "\nso=structured_outputs rf=response_format think=mandatory reasoning\n\nSuggested .env lines:"
    )
    for line in env_lines(pick_roles(models)):
        typer.echo(line)


if __name__ == "__main__":
    main()
