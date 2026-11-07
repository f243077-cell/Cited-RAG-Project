"""E8: standard vs Deep vs Agent on the same items (PHASE3_ARCHITECTURE section 11). No LLM: it reads run folders.

Coverage is measured on the passages the generator saw (`ItemRecord.retrieved`, filled from Deps.observe), so every mode is
judged on what reached the answer. For Agent mode the injection check also reads every tool argument in the trace, because a
poisoned passage could steer a search without leaking into the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from citerag.eval.metrics import full_coverage_at_k
from citerag.eval.report import NOISE_POINTS
from citerag.eval.runner import (
    WORST_AGENT_REQUESTS,
    WORST_ANSWER_REQUESTS,
    WORST_DEEP_REQUESTS,
    WORST_IMAGE_REQUESTS,
)
from citerag.eval.scoring import ANSWER_TYPES, rate
from citerag.schemas import GoldenItem, ItemRecord, Rate, RunSummary

_PAGE = re.compile(r"^(?P<doc>.+) p(?P<page>\d+)$")
TYPICAL = {"standard": 1, "deep": 3, "agent": 3}  # answer requests per item when nothing fails
WORST = {"standard": WORST_ANSWER_REQUESTS, "deep": WORST_DEEP_REQUESTS, "agent": WORST_AGENT_REQUESTS}


def parse_label(label: str) -> tuple[str, int | None]:
    """The inverse of eval.retrieval.label: "doc.pdf p3" or "notes.md"."""
    found = _PAGE.match(label)
    return (found["doc"], int(found["page"])) if found else (label, None)


def context_coverage(records: list[ItemRecord]) -> Rate:
    """multi_chunk items whose every gold part reached the generator."""
    multi = [r for r in records if r.type == "multi_chunk" and r.status == "answered"]
    covered = 0
    for record in multi:
        seen = [parse_label(label) for label in record.retrieved]
        covered += bool(seen) and full_coverage_at_k(seen, record.gold_sources, len(seen)) == 1.0
    return rate(covered, len(multi))


def tool_argument_leaks(records: list[ItemRecord]) -> int:
    """Injection items whose canary appears in any tool argument of the agent trace."""
    leaks = 0
    for record in records:
        if record.canary is None or record.trace is None:
            continue
        args = " ".join(str(step.args) for step in record.trace.steps if step.args)
        leaks += record.canary.lower() in args.lower()
    return leaks


SUPERVISOR_TYPICAL, SUPERVISOR_WORST = 5, 11  # one research handoff; agent_graph/supervisor.py for the cap


def estimate(
    items: list[GoldenItem], mode: str, skip_judge: bool, supervisor: bool = False
) -> tuple[int, int]:
    """(expected, worst case) LLM requests for one run. Image items cost the image path in standard mode."""
    typical, worst_case = dict(TYPICAL), dict(WORST)
    if supervisor:
        typical["agent"], worst_case["agent"] = SUPERVISOR_TYPICAL, SUPERVISOR_WORST
    expected = worst = 0
    for item in items:
        image = item.image_path is not None and mode == "standard"
        expected += 3 if image else typical[mode]
        worst += WORST_IMAGE_REQUESTS if image else worst_case[mode]
        if not skip_judge and item.type in ANSWER_TYPES:
            expected, worst = expected + 1, worst + 1
    return expected, worst


@dataclass(frozen=True)
class ModeColumn:
    name: str
    summary: RunSummary
    records: list[ItemRecord]


def _share(value: Rate | None) -> str:
    if value is None or value.rate is None:
        return "not measured"
    return f"{value.n} of {value.of} ({value.rate:.0%})"


def _ms(value: float | None) -> str:
    return "not measured" if value is None else f"{value / 1000:.1f} s"


def table(columns: list[ModeColumn]) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []

    def add(name: str, cell) -> None:  # noqa: ANN001 - a function of one column
        rows.append((name, [cell(c) for c in columns]))

    def answers(c: ModeColumn):  # noqa: ANN202
        return c.summary.answers

    add("Correct", lambda c: _share(answers(c).correctness if answers(c) else None))
    add("Correct or partial", lambda c: _share(answers(c).correct_or_partial if answers(c) else None))
    add("Faithful", lambda c: _share(answers(c).faithfulness if answers(c) else None))
    add("Citation validity", lambda c: _share(answers(c).citation_validity if answers(c) else None))
    add("multi_chunk context coverage", lambda c: _share(context_coverage(c.records)))
    add("Correct refusals", lambda c: _share(answers(c).correct_refusal if answers(c) else None))
    add("False refusals", lambda c: _share(answers(c).false_refusal if answers(c) else None))
    add(
        "Injection leaks in answers",
        lambda c: (
            f"{answers(c).injection_leaks} of {answers(c).injection_items}" if answers(c) else "not measured"
        ),
    )
    add("Canaries in tool arguments", lambda c: str(tool_argument_leaks(c.records)))
    add(
        "Mean LLM requests per item",
        lambda c: (
            "not measured"
            if not answers(c) or answers(c).mean_llm_requests is None
            else f"{answers(c).mean_llm_requests:.1f}"
        ),
    )
    add("Latency p50", lambda c: _ms(answers(c).latency_p50_ms if answers(c) else None))
    add("Latency p95", lambda c: _ms(answers(c).latency_p95_ms if answers(c) else None))
    add("Items done", lambda c: f"{c.summary.n_done} of {c.summary.n_items}")
    return rows


def render(
    columns: list[ModeColumn], chart: str | None = None, title: str = "E8: standard vs Deep vs Agent"
) -> str:
    unverified = max((c.summary.unverified_items for c in columns), default=0)
    head = "| Metric | " + " | ".join(c.name for c in columns) + " |"
    lines = [
        f"# {title}" + (" (provisional)" if unverified else ""),
        "",
        "Runs: " + ", ".join(f"`{c.summary.run_id}`" for c in columns) + ".",
        "",
        f"Differences under {NOISE_POINTS:.0f} points are noise. Verdicts are provisional while the judge is the answerer's model.",
    ]
    if unverified:
        lines += ["", f"**{unverified} items are not verified by a human: no number here is a result.**"]
    lines += ["", head, "|---|" + "---|" * len(columns)]
    lines += [f"| {name} | " + " | ".join(cells) + " |" for name, cells in table(columns)]
    if chart:
        lines += ["", f"![{title}]({chart})"]
    return "\n".join(lines) + "\n"
