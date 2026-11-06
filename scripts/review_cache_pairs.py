"""Review candidate question pairs and label them by hand. Makes no LLM request and loads no model.

    uv run python scripts/review_cache_pairs.py --out eval/results/cache_pairs_review.md   # read them all
    uv run python scripts/review_cache_pairs.py --label                                    # label one at a time

Labelling is the human's job: --label writes only what you type, to eval/cache_pairs.jsonl, after every answer. Run
it again to continue. Press s to leave a pair out and q to stop.

    e  equivalent: the same answer would be right for both questions, so a cache hit is safe
    n  near miss: they look alike but need different answers, so a cache hit would be wrong

`proposed` is what the drafter meant, not a label. The guard line shows what the number and negation check would do. A
pair the guard passes can only be kept apart by the threshold, so those near misses deserve the closest reading.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from citerag.cache.semantic_cache import guard_ok
from citerag.eval.cache_tools import CachePair, PairLabel, load_pairs, save_pairs

CANDIDATES = Path("eval/cache_pairs.candidates.jsonl")
LABELLED = Path("eval/cache_pairs.jsonl")
KEYS: dict[str, PairLabel] = {"e": "equivalent", "n": "near_miss"}
_WORD = re.compile(r"[\w.@/-]+")


def _words(text: str) -> list[str]:
    return [w.strip(".?").lower() for w in _WORD.findall(text)]


def differs(pair: CachePair) -> tuple[list[str], list[str]]:
    a, b = _words(pair.question_a), _words(pair.question_b)
    return [w for w in a if w not in b], [w for w in b if w not in a]


def render_pair(pair: CachePair) -> str:
    only_a, only_b = differs(pair)
    guard = "passes" if guard_ok(pair.question_a, pair.question_b) else "blocks a hit"
    return "\n".join(
        [
            f"\n# {pair.id} ({pair.kind}, proposed: {pair.proposed}, source: {pair.source or 'none'})",
            f"A: {pair.question_a}",
            f"B: {pair.question_b}",
            f"Only in A: {', '.join(only_a) or '(nothing)'}. Only in B: {', '.join(only_b) or '(nothing)'}.",
            f"Guard: {guard}.",
            f"Label: {pair.label if pair.labelled else 'not labelled'}",
        ]
    )


def label_pairs(
    pairs: list[CachePair],
    labelled: Path,
    ask: Callable[[CachePair], str],
) -> int:
    """Saves after each answer. Returns how many this call labelled."""
    done = {pair.id: pair for pair in load_pairs(labelled)} if labelled.exists() else {}
    added = 0
    for pair in pairs:
        if pair.id in done:
            continue
        answer = ask(pair).strip().lower()
        if answer == "q":
            break
        if answer not in KEYS:
            continue
        done[pair.id] = pair.model_copy(update={"labelled": True, "label": KEYS[answer]})
        save_pairs(labelled, done.values())
        added += 1
    return added


def _ask(pair: CachePair) -> str:
    typer.echo(render_pair(pair))
    return typer.prompt("e equivalent, n near miss, s skip, q quit")


def main(
    pairs: Annotated[Path, typer.Option(help="Candidate pairs")] = CANDIDATES,
    labelled: Annotated[Path, typer.Option(help="Where --label writes")] = LABELLED,
    label: Annotated[bool, typer.Option(help="Label one pair at a time")] = False,
    out: Annotated[Path | None, typer.Option(help="Write the review here instead of the console")] = None,
) -> None:
    candidates = load_pairs(pairs)
    if label:
        added = label_pairs(candidates, labelled, _ask)
        typer.echo(f"{added} labelled this time. Saved to {labelled}")
        return
    text = "\n".join(render_pair(pair) for pair in candidates)
    if out is None:
        sys.stdout.reconfigure(encoding="utf-8")
        typer.echo(text)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        typer.echo(f"{len(candidates)} pair(s) written to {out}")


if __name__ == "__main__":
    typer.run(main)
