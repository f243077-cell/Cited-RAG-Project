"""Golden-set runner. Saves after every item, so a run survives quota limits and resumes (F-EVA-9).

A run stops at once on llm_unavailable or budget_exceeded: a hanging provider would otherwise cost each remaining
item a full timeout and three daily-guard slots. Only an unusable reply for one item moves on to the next.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from citerag.config import Settings
from citerag.errors import (
    BudgetExceededError,
    LLMUnavailableError,
    StructuredOutputError,
    ValidationFailedError,
)
from citerag.eval.judge import judge_answer
from citerag.eval.retrieval import RetrievalReport, label, score_retrieval
from citerag.eval.scoring import needs_judge
from citerag.prompts import PROMPTS_DIR, load_prompt
from citerag.rag.context import Chunk, render_passages
from citerag.rag.pipeline import Deps, answer_question
from citerag.rag.verify import Passage
from citerag.schemas import ChatRequest, GoldenItem, ItemRecord, RetrievalScores

WORST_ANSWER_REQUESTS = 3  # rewrite, generate, regenerate; a first turn has no condense
WORST_IMAGE_REQUESTS = 5  # adds image analysis and condense
WORST_DEEP_REQUESTS = 5  # three agent tasks, rewrite, regenerate
WORST_AGENT_REQUESTS = 8  # six plan, act and reflect calls, synthesis, one citation retry
_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
_SECRETS = {"openrouter_api_key", "gemini_api_key", "citerag_api_key"}
_FINGERPRINT = ("model_fast", "model_strong", "model_judge", "provider_strong", "provider_judge")


@dataclass
class RunResult:
    status: str
    reason: str | None
    records: list[ItemRecord]
    requests: int


def to_scores(report: RetrievalReport) -> RetrievalScores:
    return RetrievalScores(**report.model_dump(include=set(RetrievalScores.model_fields)))


def score_modes(
    items: list[GoldenItem],
    rank: Callable[[str], list[Chunk]],
    cosine: Callable[[str], float],
    reranked: Callable[[str], list[Chunk]] | None,
    top_k_context: int,
) -> tuple[RetrievalReport, RetrievalReport | None]:
    before = score_retrieval(items, rank, cosine)
    if reranked is None:
        return before, None
    ks = tuple(k for k in before.ks if k <= top_k_context)
    return before, score_retrieval(items, reranked, cosine, ks=ks, mrr_k=ks[-1])


def _git_commit() -> str:
    try:
        found = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return found.stdout.strip()


def _prompt_versions() -> dict[str, str]:
    return {path.stem: load_prompt(path.stem).version for path in PROMPTS_DIR.glob("*.md")}


def run_config(settings: Settings, golden: Path, experiment: str | None) -> dict:
    """Everything needed to repeat the run. API keys stay out."""
    return {
        "settings": settings.model_dump(mode="json", exclude=_SECRETS),
        "git_commit": _git_commit(),
        "prompts": _prompt_versions(),
        "golden": str(golden),
        "golden_sha256": hashlib.sha256(golden.read_bytes()).hexdigest(),
        "experiment": experiment,
    }


def _fingerprint(config: dict) -> dict:
    models = {key: config["settings"][key] for key in _FINGERPRINT}
    return {**models, "prompts": config["prompts"], "golden_sha256": config["golden_sha256"]}


def start_or_resume(run_dir: Path, config: dict) -> None:
    """Items answered by another model or prompt must not mix into one run."""
    path = run_dir / "config.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if _fingerprint(stored) != _fingerprint(config):
            raise ValidationFailedError(
                f"{run_dir.name} used other models, prompts or golden items; start a new run"
            )
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_records(run_dir: Path) -> dict[str, ItemRecord]:
    path = run_dir / "items.jsonl"
    if not path.exists():
        return {}
    records = [
        ItemRecord.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line
    ]
    return {record.id: record for record in records}


def save_records(run_dir: Path, records: dict[str, ItemRecord]) -> None:
    """Write then rename, so a crash never leaves half a file."""
    temp = run_dir / "items.tmp"
    temp.write_text("\n".join(r.model_dump_json() for r in records.values()) + "\n", encoding="utf-8")
    os.replace(temp, run_dir / "items.jsonl")


def _base_record(item: GoldenItem, **fields: object) -> ItemRecord:
    return ItemRecord(
        id=item.id,
        type=item.type,
        question=item.question,
        gold_answer=item.gold_answer,
        gold_sources=item.gold_sources,
        canary=item.canary,
        **fields,
    )


def _image_fields(item: GoldenItem) -> dict[str, str]:
    if item.image_path is None:
        return {}
    path = Path(item.image_path)
    return {
        "image_b64": base64.b64encode(path.read_bytes()).decode("ascii"),
        "image_media_type": _MEDIA_TYPES[path.suffix.lower()],
    }


async def _answer_item(item: GoldenItem, deps: Deps, run_id: str, mode: str) -> ItemRecord:
    seen: dict[str, str | list[str]] = {"context": "", "retrieved": []}

    def observe(passages: list[Passage], chunks: list[Chunk]) -> None:
        seen.update(context=render_passages(passages), retrieved=[label(c) for c in chunks])

    deps.observe = observe
    session = f"eval-{run_id}-{item.id}-{uuid.uuid4().hex[:8]}"  # no history crosses items or resumes
    request = ChatRequest(
        session_id=session, message=item.question, use_cache=False, mode=mode, **_image_fields(item)
    )
    response = await answer_question(request, deps)
    return _base_record(
        item,
        status="answered",
        answer=response.answer,
        context=seen["context"],
        retrieved=seen["retrieved"],
        warnings=response.warnings,
        usage=response.usage,
    )


async def run_items(
    items: list[GoldenItem],
    deps: Deps,
    run_dir: Path,
    max_requests: int | None = None,
    skip_judge: bool = False,
    mode: str = "standard",
) -> RunResult:
    records = load_records(run_dir)
    spent = 0

    def result(status: str, reason: str | None = None) -> RunResult:
        save_records(run_dir, records) if records else None
        return RunResult(status, reason, [records[i.id] for i in items if i.id in records], spent)

    def over_cap(next_cost: int) -> bool:
        return max_requests is not None and spent + next_cost > max_requests

    for item in items:
        record = records.get(item.id)
        if record is None or record.status == "error":
            worst = WORST_IMAGE_REQUESTS if item.image_path else WORST_ANSWER_REQUESTS
            worst = {"deep": WORST_DEEP_REQUESTS, "agent": WORST_AGENT_REQUESTS}.get(mode, worst)
            if over_cap(worst):
                return result("paused", f"--max-requests {max_requests} reached")
            try:
                record = await _answer_item(item, deps, run_dir.name, mode)
            except BudgetExceededError as exc:
                return result("paused", exc.message)
            except LLMUnavailableError as exc:
                return result("stopped", exc.message)
            except StructuredOutputError as exc:
                record = _base_record(item, status="error", error_code=exc.code)
            spent += record.usage.llm_requests if record.usage else 0
            records[item.id] = record
            save_records(run_dir, records)

        if skip_judge or not needs_judge(record):
            continue
        if over_cap(1):
            return result("paused", f"--max-requests {max_requests} reached")
        try:
            verdict = await judge_answer(deps.llm, record, f"eval-{run_dir.name}-{item.id}")
        except BudgetExceededError as exc:
            return result("paused", exc.message)
        except LLMUnavailableError as exc:
            return result("stopped", f"judge: {exc.message}")
        except StructuredOutputError as exc:
            record.judge_error = exc.code
        else:
            record.verdict, record.judge_error = verdict.parsed, None
            record.judge_requests += verdict.requests
            spent += verdict.requests
        save_records(run_dir, records)
    return result("complete")
