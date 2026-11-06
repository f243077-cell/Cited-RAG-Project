"""Cache threshold sweep (F-CAC-8, experiment E5). Zero LLM requests: embeddings are local.

Only pairs a human labelled count. The sweep proposes a threshold and edits nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ValidationError, model_validator

from citerag.cache.semantic_cache import _unit, guard_ok
from citerag.errors import ValidationFailedError
from citerag.eval.metrics import pick_threshold, threshold_sweep
from citerag.eval.report import RESULTS_DIR

PairLabel = Literal["equivalent", "near_miss"]
PairKind = Literal["paraphrase", "number", "entity", "unit", "negation", "scope"]
MIN_PER_CLASS = 30  # F-CAC-8: fewer than this and one pair moves the false-hit rate past the 2% cap
SWEEP_FILE = RESULTS_DIR / "cache_sweep" / "sweep.json"
CANDIDATES = Path("eval/cache_pairs.candidates.jsonl")
LABEL_COMMAND = "uv run python scripts/review_cache_pairs.py --label"


class CachePair(BaseModel):
    id: str
    question_a: str
    question_b: str
    proposed: PairLabel  # what the drafter meant; a hint for the reviewer, never used as a label
    kind: PairKind
    source: str | None = None  # golden item the pair came from
    labelled: bool = False
    label: PairLabel | None = None

    @model_validator(mode="after")
    def _label_goes_with_labelled(self) -> CachePair:
        if self.labelled != (self.label is not None):
            raise ValueError("label must be set exactly when labelled is true")
        return self


class SweepRow(BaseModel):
    tau: float
    precision: float
    recall: float
    false_hit_rate: float


class PairScore(BaseModel):
    id: str
    label: PairLabel
    kind: PairKind
    similarity: float
    guard_ok: bool
    question_a: str
    question_b: str


class SweepResult(BaseModel):
    n_equivalent: int
    n_near_miss: int
    enough_pairs: bool
    max_false_hit_rate: float
    recommended: float | None
    rows: list[SweepRow]  # what the cache does: similarity at or above tau AND the guard passes
    rows_without_guard: list[SweepRow]
    worst_near_misses: list[PairScore]  # the pairs that decide the threshold: read these first


def load_pairs(path: Path) -> list[CachePair]:
    if not path.exists():
        raise ValidationFailedError(
            f"{path} not found. Candidates are in {CANDIDATES}: label them with {LABEL_COMMAND}"
        )
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        return [CachePair.model_validate_json(line) for line in lines]
    except ValidationError as exc:
        raise ValidationFailedError(f"{path}: {exc}") from exc


def save_pairs(path: Path, pairs: Iterable[CachePair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(pair.model_dump_json() + "\n" for pair in pairs), encoding="utf-8")


def _rows(sims: np.ndarray, equivalent: np.ndarray) -> list[SweepRow]:
    return [SweepRow(**row) for row in threshold_sweep(sims, equivalent).to_dict("records")]


def score_pair(pair: CachePair, embed: Callable[[str], list[float]]) -> PairScore:
    similarity = float(_unit(np.asarray(embed(pair.question_a))) @ _unit(np.asarray(embed(pair.question_b))))
    return PairScore(
        id=pair.id,
        label=pair.label,
        kind=pair.kind,
        similarity=similarity,
        guard_ok=guard_ok(pair.question_a, pair.question_b),
        question_a=pair.question_a,
        question_b=pair.question_b,
    )


def sweep_pairs(
    pairs: list[CachePair], embed: Callable[[str], list[float]], max_false_hit_rate: float = 0.02
) -> SweepResult:
    """A pair the guard rejects can never hit, so it counts as similarity -1 in the rows the cache follows."""
    labelled = [pair for pair in pairs if pair.labelled]
    if not labelled:
        raise ValidationFailedError(
            f"No labelled pairs, so there is nothing to sweep. Label the candidates with {LABEL_COMMAND}"
        )
    scores = [score_pair(pair, embed) for pair in labelled]
    sims = np.array([s.similarity for s in scores])
    equivalent = np.array([s.label == "equivalent" for s in scores])
    guarded = np.where([s.guard_ok for s in scores], sims, -1.0)
    n_equivalent, n_near_miss = int(equivalent.sum()), int((~equivalent).sum())
    rows = _rows(guarded, equivalent)
    frame = threshold_sweep(guarded, equivalent)
    worst = sorted((s for s in scores if s.label == "near_miss"), key=lambda s: -s.similarity)
    return SweepResult(
        n_equivalent=n_equivalent,
        n_near_miss=n_near_miss,
        enough_pairs=min(n_equivalent, n_near_miss) >= MIN_PER_CLASS,
        max_false_hit_rate=max_false_hit_rate,
        recommended=pick_threshold(frame, max_false_hit_rate) if n_near_miss and n_equivalent else None,
        rows=rows,
        rows_without_guard=_rows(sims, equivalent),
        worst_near_misses=worst[:10],
    )


def save_sweep(result: SweepResult, path: Path = SWEEP_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def load_sweep(path: Path = SWEEP_FILE) -> SweepResult | None:
    return SweepResult.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None
