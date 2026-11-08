"""Move golden items you have reviewed into eval/golden/golden.jsonl with verified: true. For the human reviewer only.

    uv run python scripts/review_golden.py --out eval/results/golden_review.md   # read first
    uv run python scripts/accept_golden.py --ids a-001,a-002,m-001              # then accept what you checked

Fix a question, answer or source in eval/golden/candidates.jsonl before accepting it: the item is copied as it stands, with
only `verified` set. Nothing here runs by itself, and no other code path sets `verified`. An id already in golden.jsonl is
replaced, so a corrected item can be accepted again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from citerag.eval.golden import load_golden
from citerag.schemas import GoldenItem


def accept(
    candidates: list[GoldenItem], golden: list[GoldenItem], ids: list[str], reviewed_by: str = "human"
) -> list[GoldenItem]:
    by_id = {item.id: item for item in candidates}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise typer.BadParameter(f"not in the candidates: {', '.join(unknown)}")
    kept = {item.id: item for item in golden if item.id not in ids}
    accepted = {i: by_id[i].model_copy(update={"verified": True, "reviewed_by": reviewed_by}) for i in ids}
    return [*kept.values(), *accepted.values()]


def main(
    ids: Annotated[str, typer.Option(help="Comma-separated ids you reviewed, e.g. a-001,m-002")],
    candidates: Annotated[Path, typer.Option()] = Path("eval/golden/candidates.jsonl"),
    golden: Annotated[Path, typer.Option()] = Path("eval/golden/golden.jsonl"),
    reviewed_by: Annotated[
        str, typer.Option(help="human, or claude for an AI review (reports say so)")
    ] = "human",
) -> None:
    if reviewed_by not in ("human", "claude"):
        raise typer.BadParameter("--reviewed-by must be human or claude")
    chosen = [i.strip() for i in ids.split(",") if i.strip()]
    if not chosen:
        raise typer.BadParameter("give at least one id")
    current = load_golden(golden) if golden.exists() else []
    items = accept(load_golden(candidates), current, chosen, reviewed_by)
    golden.write_text(
        "".join(item.model_dump_json(exclude_none=True) + "\n" for item in items), encoding="utf-8"
    )
    typer.echo(f"{len(chosen)} accepted ({reviewed_by}); {golden} now holds {len(items)} verified item(s).")


if __name__ == "__main__":
    typer.run(main)
