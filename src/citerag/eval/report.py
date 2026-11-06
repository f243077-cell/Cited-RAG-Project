"""Reads run folders under eval/results/ for the CLI and the API. No LLM, no index."""

from __future__ import annotations

import re
from pathlib import Path

from citerag.errors import NotFoundError
from citerag.eval.runner import load_records
from citerag.schemas import RunDetail, RunSummary

RESULTS_DIR = Path("eval/results")
_RUN_ID = re.compile(r"[\w.-]+")
NOISE_POINTS = 3.0  # with about 40 items one item is 2.5 points


def run_path(run_id: str, root: Path = RESULTS_DIR) -> Path:
    """A run id comes from a URL, so it must stay one plain folder name."""
    path = root / run_id
    if not _RUN_ID.fullmatch(run_id) or set(run_id) == {"."} or not (path / "summary.json").is_file():
        raise NotFoundError(f"no run {run_id}")
    return path


def load_summary(run_id: str, root: Path = RESULTS_DIR) -> RunSummary:
    return RunSummary.model_validate_json(
        (run_path(run_id, root) / "summary.json").read_text(encoding="utf-8")
    )


def load_detail(run_id: str, root: Path = RESULTS_DIR) -> RunDetail:
    records = load_records(run_path(run_id, root))
    return RunDetail(summary=load_summary(run_id, root), items=list(records.values()))


def list_runs(root: Path = RESULTS_DIR) -> list[RunSummary]:
    """Newest first. Folders from the older `eval retrieval` command have another shape and are left out."""
    runs = []
    if root.is_dir():
        for path in sorted(root.iterdir(), reverse=True):
            try:
                runs.append(load_summary(path.name, root))
            except (NotFoundError, ValueError):
                continue
    return runs


def compare_rows(a: RunSummary, b: RunSummary) -> list[tuple[str, float | None, float | None, float | None]]:
    """(metric, A, B, B minus A in points). None marks a metric one of the runs did not measure."""
    left, right = a.headline(), b.headline()
    rows = []
    for name in [*left, *(k for k in right if k not in left)]:
        x, y = left.get(name), right.get(name)
        rows.append((name, x, y, None if x is None or y is None else (y - x) * 100))
    return rows
