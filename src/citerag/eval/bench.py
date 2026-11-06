"""Cache benchmark (F-CAC-9, experiment E6): replay a workload with the cache off, then on, and compare.

Both passes save after every item, so a run resumes across free-tier days. The cache lives in the run folder, so the
benchmark starts empty and never touches the user's cache. A cache hit reuses the verdict of the answer it served when
the gold answer is the same, because that is the same text graded against the same reference: a hit that served the
wrong answer has another gold answer, so it still costs a judge request.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from citerag.errors import (
    BudgetExceededError,
    LLMUnavailableError,
    StructuredOutputError,
    ValidationFailedError,
)
from citerag.eval.judge import judge_answer
from citerag.eval.metrics import percentile
from citerag.eval.runner import WORST_ANSWER_REQUESTS
from citerag.eval.scoring import needs_judge
from citerag.rag.context import Chunk, render_passages
from citerag.rag.pipeline import Deps, answer_question
from citerag.rag.verify import Passage
from citerag.schemas import ChatRequest, ItemRecord, JudgeVerdict, Rate

PassName = Literal["off", "on"]
HIT_P95_TARGET_MS = 300  # G4
MIN_REPEAT_SHARE = 0.40  # G4: the hit-rate target is defined on a workload with this many paraphrased repeats


class WorkloadItem(BaseModel):
    id: str
    question: str
    gold_answer: str | None = None
    repeat_of: str | None = None  # id of the earlier item this one paraphrases


class BenchRecord(ItemRecord):
    repeat_of: str | None = None
    cache_hit: bool = False
    similarity: float | None = None
    matched_query: str | None = None


class PassSummary(BaseModel):
    name: PassName
    n_items: int
    n_done: int
    hits: int
    hit_rate: Rate
    errors: int
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    hit_latency_p95_ms: float | None = None
    llm_requests: int  # answers only: the judge is the measurement, not the system
    judge_requests: int
    input_tokens: int
    output_tokens: int
    shadow_usd: float
    correct: Rate  # judged correct, out of items answered or refused


class BenchSummary(BaseModel):
    run_id: str
    status: Literal["complete", "paused", "stopped"]
    reason: str | None = None
    workload: str
    threshold: float
    repeat_share: float
    passes: dict[str, PassSummary]
    requests_used: int
    request_reduction: float | None = None
    correctness_change_points: float | None = None
    false_hits: list[str] = []  # hits judged incorrect: the failure this benchmark exists to catch


@dataclass
class Spend:
    limit: int | None
    used: int = 0

    def over(self, next_cost: int) -> bool:
        return self.limit is not None and self.used + next_cost > self.limit


@dataclass
class PassState:
    records: dict[str, BenchRecord] = field(default_factory=dict)
    contexts: dict[str, str] = field(default_factory=dict)  # by standalone question, so a hit can be judged
    verdicts: dict[tuple[str, str | None], JudgeVerdict] = field(default_factory=dict)


def load_workload(path: Path) -> list[WorkloadItem]:
    if not path.exists():
        raise ValidationFailedError(
            f"{path} not found. Build it from labelled pairs with: uv run python scripts/make_workload.py"
        )
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        return [WorkloadItem.model_validate_json(line) for line in lines]
    except ValidationError as exc:
        raise ValidationFailedError(f"{path}: {exc}") from exc


def repeat_share(items: list[WorkloadItem]) -> float:
    return sum(item.repeat_of is not None for item in items) / len(items) if items else 0.0


def estimate_requests(items: list[WorkloadItem], skip_judge: bool = False) -> tuple[int, int]:
    """(expected, worst case). Expected assumes every repeat hits and each answer costs one request."""
    originals = sum(item.repeat_of is None for item in items)
    answers = len(items) + originals
    expected = answers * (1 if skip_judge else 2)
    worst = len(items) * 2 * (WORST_ANSWER_REQUESTS + (0 if skip_judge else 1))
    return expected, worst


def _pass_path(run_dir: Path, name: PassName) -> Path:
    return run_dir / f"pass_{name}.jsonl"


def _load_state(run_dir: Path, name: PassName) -> PassState:
    path = _pass_path(run_dir, name)
    state = PassState()
    if path.exists():
        rows = [
            BenchRecord.model_validate_json(x) for x in path.read_text(encoding="utf-8").splitlines() if x
        ]
        state.records = {row.id: row for row in rows}
        for row in rows:
            _remember(state, row)
    return state


def _save(run_dir: Path, name: PassName, state: PassState) -> None:
    text = "".join(record.model_dump_json() + "\n" for record in state.records.values())
    _pass_path(run_dir, name).write_text(text, encoding="utf-8")


def _remember(state: PassState, record: BenchRecord) -> None:
    if record.context and not record.cache_hit:
        state.contexts[record.question] = record.context
    if record.answer is not None and record.verdict is not None:
        state.verdicts[(record.answer.answer, record.gold_answer)] = record.verdict


async def _answer(item: WorkloadItem, deps: Deps, use_cache: bool, state: PassState) -> BenchRecord:
    seen: dict[str, str] = {"context": ""}

    def observe(passages: list[Passage], _: list[Chunk]) -> None:
        seen["context"] = render_passages(passages)

    deps.observe = observe
    session = f"bench-{item.id}-{uuid.uuid4().hex[:8]}"  # no history, so no condense request
    response = await answer_question(
        ChatRequest(session_id=session, message=item.question, use_cache=use_cache), deps
    )
    matched = response.cache.matched_query
    return BenchRecord(
        id=item.id,
        type="answerable",
        question=item.question,
        gold_answer=item.gold_answer,
        status="answered",
        answer=response.answer,
        context=seen["context"] or state.contexts.get(matched or "", ""),
        warnings=response.warnings,
        usage=response.usage,
        repeat_of=item.repeat_of,
        cache_hit=response.cache.hit,
        similarity=response.cache.similarity,
        matched_query=matched,
    )


async def run_pass(
    name: PassName,
    items: list[WorkloadItem],
    deps: Deps,
    run_dir: Path,
    spend: Spend,
    skip_judge: bool,
) -> tuple[str, str | None]:
    state = _load_state(run_dir, name)
    for item in items:
        record = state.records.get(item.id)
        if record is None or record.status == "error":
            if spend.over(WORST_ANSWER_REQUESTS):
                return "paused", f"--max-requests {spend.limit} reached"
            try:
                record = await _answer(item, deps, name == "on", state)
            except BudgetExceededError as exc:
                return "paused", exc.message
            except LLMUnavailableError as exc:
                return "stopped", exc.message
            except StructuredOutputError as exc:
                record = BenchRecord(
                    id=item.id,
                    type="answerable",
                    question=item.question,
                    gold_answer=item.gold_answer,
                    status="error",
                    error_code=exc.code,
                    repeat_of=item.repeat_of,
                )
            spend.used += record.usage.llm_requests if record.usage else 0
            state.records[item.id] = record
            _remember(state, record)
            _save(run_dir, name, state)

        if skip_judge or not needs_judge(record):
            continue
        reused = state.verdicts.get((record.answer.answer, record.gold_answer))
        if reused is not None:
            record.verdict = reused
            _save(run_dir, name, state)
            continue
        if spend.over(1):
            return "paused", f"--max-requests {spend.limit} reached"
        try:
            verdict = await judge_answer(deps.llm, record, f"bench-{run_dir.name}-{name}-{item.id}")
        except BudgetExceededError as exc:
            return "paused", exc.message
        except LLMUnavailableError as exc:
            return "stopped", f"judge: {exc.message}"
        except StructuredOutputError as exc:
            record.judge_error = exc.code
        else:
            record.verdict, record.judge_error = verdict.parsed, None
            record.judge_requests += verdict.requests
            spend.used += verdict.requests
            _remember(state, record)
        _save(run_dir, name, state)
    return "complete", None


async def run_both(
    items: list[WorkloadItem], off: Deps, on: Deps, run_dir: Path, spend: Spend, skip_judge: bool
) -> tuple[str, str | None]:
    """Cache off first: that pass sets the baseline, and the second is only comparable once it is whole."""
    status, reason = await run_pass("off", items, off, run_dir, spend, skip_judge)
    if status != "complete":
        return status, reason
    return await run_pass("on", items, on, run_dir, spend, skip_judge)


def _rate(n: int, of: int) -> Rate:
    return Rate(n=n, of=of, rate=n / of if of else None)


def summarize_pass(name: PassName, items: list[WorkloadItem], records: dict[str, BenchRecord]) -> PassSummary:
    done = [records[item.id] for item in items if item.id in records]
    used = [r.usage for r in done if r.usage is not None]
    latencies = [u.latency_ms.total for u in used]
    hits = [r for r in done if r.cache_hit]
    hit_latencies = [r.usage.latency_ms.total for r in hits if r.usage is not None]
    answered = [r for r in done if r.status == "answered"]
    correct = sum(r.verdict is not None and r.verdict.correctness == "correct" for r in answered)
    return PassSummary(
        name=name,
        n_items=len(items),
        n_done=len(done),
        hits=len(hits),
        hit_rate=_rate(len(hits), len(done)),
        errors=sum(r.status == "error" for r in done),
        latency_p50_ms=percentile(latencies, 50) if latencies else None,
        latency_p95_ms=percentile(latencies, 95) if latencies else None,
        hit_latency_p95_ms=percentile(hit_latencies, 95) if hit_latencies else None,
        llm_requests=sum(u.llm_requests for u in used),
        judge_requests=sum(r.judge_requests for r in done),
        input_tokens=sum(u.input_tokens for u in used),
        output_tokens=sum(u.output_tokens for u in used),
        shadow_usd=sum(u.est_cost_usd for u in used),
        correct=_rate(correct, len(answered)),
    )


def build_summary(
    run_id: str,
    status: str,
    reason: str | None,
    workload: Path,
    threshold: float,
    items: list[WorkloadItem],
    run_dir: Path,
    spent: int,
) -> BenchSummary:
    states = {name: _load_state(run_dir, name) for name in ("off", "on")}
    off, on = (summarize_pass(name, items, states[name].records) for name in ("off", "on"))
    finished = off.n_done == on.n_done == len(items)
    reduction = 1 - on.llm_requests / off.llm_requests if finished and off.llm_requests else None
    change = None
    if finished and off.correct.rate is not None and on.correct.rate is not None:
        change = (on.correct.rate - off.correct.rate) * 100
    wrong = [
        r.id
        for r in states["on"].records.values()
        if r.cache_hit and r.verdict is not None and r.verdict.correctness == "incorrect"
    ]
    return BenchSummary(
        run_id=run_id,
        status=status,
        reason=reason,
        workload=str(workload),
        threshold=threshold,
        repeat_share=repeat_share(items),
        passes={"off": off, "on": on},
        requests_used=spent,
        request_reduction=reduction,
        correctness_change_points=change,
        false_hits=wrong,
    )


def _ms(value: float | None) -> str:
    return "not measured" if value is None else f"{value:,.0f} ms"


def _share(rate: Rate) -> str:
    return f"{rate.n} of {rate.of}" + ("" if rate.rate is None else f" ({rate.rate:.0%})")


def render_report(summary: BenchSummary, chart: str | None = None) -> str:
    """Provisional until the workload's gold answers and pairs are human-verified and the judge differs."""
    off, on = summary.passes["off"], summary.passes["on"]
    reduction = summary.request_reduction
    change = summary.correctness_change_points
    lines = [
        "# E6: does the cache pay off? (provisional)",
        "",
        f"Run `{summary.run_id}`. Threshold {summary.threshold}. Workload `{summary.workload}`: "
        f"{off.n_items} questions, {summary.repeat_share:.0%} paraphrased repeats.",
        "",
        "Same model as answerer judges these answers unless `MODEL_JUDGE` was changed, so correctness is provisional.",
        "",
        "| Measure | Cache off | Cache on |",
        "|---|---|---|",
        f"| Hit rate | {_share(off.hit_rate)} | {_share(on.hit_rate)} |",
        f"| Latency p50 | {_ms(off.latency_p50_ms)} | {_ms(on.latency_p50_ms)} |",
        f"| Latency p95 | {_ms(off.latency_p95_ms)} | {_ms(on.latency_p95_ms)} |",
        f"| LLM requests | {off.llm_requests} | {on.llm_requests} |",
        f"| Tokens in, out | {off.input_tokens}, {off.output_tokens} | {on.input_tokens}, {on.output_tokens} |",
        f"| Shadow dollars | {off.shadow_usd:.4f} | {on.shadow_usd:.4f} |",
        f"| Judged correct | {_share(off.correct)} | {_share(on.correct)} |",
        "",
        "| Target (G4) | Goal | Result |",
        "|---|---|---|",
        f"| Hit rate | at least 30% | {_share(on.hit_rate)} |",
        f"| LLM-request reduction | at least 25% | {'not measured' if reduction is None else f'{reduction:.0%}'} |",
        f"| Correctness change | at least -1 point | {'not measured' if change is None else f'{change:+.1f} points'} |",
        f"| Hit latency p95 | at most {HIT_P95_TARGET_MS} ms | {_ms(on.hit_latency_p95_ms)} |",
        "",
        f"False hits, judged incorrect: {', '.join(summary.false_hits) or 'none'}.",
    ]
    if summary.repeat_share < MIN_REPEAT_SHARE:
        lines += [
            "",
            f"The workload has under {MIN_REPEAT_SHARE:.0%} repeats, so G4's hit-rate target does not apply.",
        ]
    if chart:
        lines += ["", f"![Cache off vs on]({chart})"]
    return "\n".join(lines) + "\n"
