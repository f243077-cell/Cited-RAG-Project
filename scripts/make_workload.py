"""Build eval/workload.jsonl for bench-cache from golden items and human-labelled pairs. No LLM, no model.

    uv run python scripts/make_workload.py

Each answerable or multi_chunk golden item goes in once, then each equivalent pair adds its paraphrase as a repeat of
that item, with the same gold answer. Only pairs a human labelled equivalent count, because the paraphrase inherits
the gold answer. Near misses stay out: they have no gold answer to grade against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from citerag.eval.bench import MIN_REPEAT_SHARE, WorkloadItem, repeat_share
from citerag.eval.cache_tools import CachePair, load_pairs
from citerag.eval.golden import load_golden
from citerag.schemas import GoldenItem

ANSWERED = ("answerable", "multi_chunk")


def build_workload(golden: list[GoldenItem], pairs: list[CachePair]) -> list[WorkloadItem]:
    sources = {item.id: item for item in golden if item.type in ANSWERED}
    items = [WorkloadItem(id=i.id, question=i.question, gold_answer=i.gold_answer) for i in sources.values()]
    repeats = 0
    for pair in pairs:
        if pair.label != "equivalent" or pair.source not in sources:
            continue
        repeats += 1
        source = sources[pair.source]
        items.append(
            WorkloadItem(
                id=f"{pair.source}-r{repeats:03d}",
                question=pair.question_b,
                gold_answer=source.gold_answer,
                repeat_of=pair.source,
            )
        )
    return items


def main(
    golden: Annotated[Path, typer.Option()] = Path("eval/golden/candidates.jsonl"),
    pairs: Annotated[Path, typer.Option()] = Path("eval/cache_pairs.jsonl"),
    out: Annotated[Path, typer.Option()] = Path("eval/workload.jsonl"),
) -> None:
    labelled = [pair for pair in load_pairs(pairs) if pair.labelled]
    items = build_workload(load_golden(golden), labelled)
    if not any(item.repeat_of for item in items):
        typer.echo(
            f"No labelled equivalent pairs in {pairs}, so there are no repeats and nothing is written."
        )
        raise typer.Exit(1)
    out.write_text("".join(item.model_dump_json() + "\n" for item in items), encoding="utf-8")
    typer.echo(f"{len(items)} questions, {repeat_share(items):.0%} repeats, written to {out}")
    if repeat_share(items) < MIN_REPEAT_SHARE:
        typer.echo(f"Under {MIN_REPEAT_SHARE:.0%} repeats: G4's hit-rate target needs at least that many.")


if __name__ == "__main__":
    typer.run(main)
