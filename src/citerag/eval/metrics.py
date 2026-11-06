"""Retrieval metrics and cache-threshold analysis (numpy + pandas)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from citerag.schemas import GoldenSource


def is_hit(doc: str, page: int | None, gold: list[GoldenSource]) -> bool:
    """A retrieved chunk is relevant if it comes from a gold document and (when given) the gold page."""
    return any(g.doc == doc and (g.page is None or g.page == page) for g in gold)


def hit_at_k(relevance: list[bool], k: int) -> float:
    """relevance[i] says whether the i-th retrieved chunk is relevant. 1.0 if any of the top k is."""
    return float(any(relevance[:k]))


def parts_covered(
    retrieved: list[tuple[str, int | None]], gold: list[GoldenSource], k: int
) -> dict[int, bool]:
    """retrieved = [(doc, page), ...] in rank order. Returns {part: True if any top-k chunk matches a source of that part}."""
    top = retrieved[:k]
    parts: dict[int, bool] = {}
    for g in gold:
        parts[g.part] = parts.get(g.part, False) or any(is_hit(d, p, [g]) for d, p in top)
    return parts


def coverage_at_k(retrieved: list[tuple[str, int | None]], gold: list[GoldenSource], k: int) -> float:
    """Share of gold parts found in the top k (0.0 to 1.0). With one part this equals hit@k."""
    covered = parts_covered(retrieved, gold, k)
    return sum(covered.values()) / len(covered) if covered else 0.0


def full_coverage_at_k(retrieved: list[tuple[str, int | None]], gold: list[GoldenSource], k: int) -> float:
    """1.0 only if EVERY gold part is found in the top k. The headline metric for multi_chunk items."""
    covered = parts_covered(retrieved, gold, k)
    return float(bool(covered) and all(covered.values()))


def reciprocal_rank(relevance: list[bool], k: int) -> float:
    for i, rel in enumerate(relevance[:k], start=1):
        if rel:
            return 1.0 / i
    return 0.0


def percentile(values: list[float] | np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else float("nan")


def threshold_sweep(
    sims: np.ndarray, is_equivalent: np.ndarray, taus: np.ndarray | None = None
) -> pd.DataFrame:
    """For labelled question pairs: what would each cache threshold do?
    false_hit_rate = share of NON-equivalent pairs wrongly served from cache (the dangerous error)."""
    sims = np.asarray(sims, dtype=float)
    eq = np.asarray(is_equivalent, dtype=bool)
    taus = np.round(np.arange(0.70, 1.0001, 0.01), 3) if taus is None else taus
    rows = []
    for t in taus:
        pred = sims >= t
        tp, fp = int((pred & eq).sum()), int((pred & ~eq).sum())
        fn, tn = int((~pred & eq).sum()), int((~pred & ~eq).sum())
        rows.append(
            {
                "tau": float(t),
                "precision": tp / (tp + fp) if tp + fp else 1.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "false_hit_rate": fp / (fp + tn) if fp + tn else 0.0,
            }
        )
    return pd.DataFrame(rows)


def pick_threshold(sweep: pd.DataFrame, max_false_hit_rate: float = 0.02) -> float | None:
    """Highest-recall threshold whose false-hit rate stays under the cap (ties: the stricter one).
    Returns None if no threshold is both safe and useful (recall > 0): then fix the embeddings or the guard."""
    ok = sweep[(sweep.false_hit_rate <= max_false_hit_rate) & (sweep.recall > 0)]
    if ok.empty:
        return None
    return float(ok.sort_values(["recall", "tau"], ascending=[False, False]).iloc[0].tau)


def estimate_cost_usd(
    model: str, input_tokens: int, output_tokens: int, pricing: dict[str, dict[str, float]]
) -> float:
    """SHADOW cost: what these tokens would cost on a paid model. Free (:free) models really cost 0, so models that
    are not in the table fall back to the "_reference" entry (the chosen paid reference model).
    pricing = {"model-id": {"input_per_mtok": x, "output_per_mtok": y}, "_reference": {...}} from configs/pricing.yaml."""
    p = pricing.get(model) or pricing["_reference"]
    return (input_tokens * p["input_per_mtok"] + output_tokens * p["output_per_mtok"]) / 1_000_000
