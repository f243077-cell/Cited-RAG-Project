"""Read-only access to saved evaluation runs for the Evaluation page (F-UI-4)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from citerag.api.deps import require_api_key
from citerag.eval.report import list_runs, load_detail
from citerag.schemas import RunDetail, RunSummary

router = APIRouter(prefix="/v1/eval", dependencies=[Depends(require_api_key)])


@router.get("/runs")
def runs() -> list[RunSummary]:
    return list_runs()


@router.get("/runs/{run_id}")
def run_detail(run_id: str) -> RunDetail:
    return load_detail(run_id)
