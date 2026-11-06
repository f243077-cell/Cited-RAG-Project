"""Filtering and role picks of scripts/pick_free_models.py, on a small synthetic model list (no network)."""

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "pick_free_models.py"
_SPEC = importlib.util.spec_from_file_location("pick_free_models", _PATH)
pfm = importlib.util.module_from_spec(_SPEC)
sys.modules["pick_free_models"] = pfm  # dataclasses look their module up here
_SPEC.loader.exec_module(pfm)


def raw(mid, prompt="0", completion="0", inputs=("text",), outputs=("text",), params=(), **extra):
    return {
        "id": mid,
        "context_length": extra.pop("context_length", 1000),
        "architecture": {"input_modalities": list(inputs), "output_modalities": list(outputs)},
        "pricing": {"prompt": prompt, "completion": completion},
        "supported_parameters": list(params),
        **extra,
    }


FULL = ("structured_outputs", "response_format", "tools", "temperature")
LIST = [
    raw("vendor/paid", prompt="0.000002", completion="0.00001", params=FULL),
    raw("vendor/half-free:free", completion="0.00001", params=FULL),
    raw("openrouter/free", inputs=("text", "image"), params=FULL),
    raw("vendor/music:free", outputs=("text", "audio"), params=("response_format",)),
    raw("stealth/no-suffix", params=FULL, context_length=9_000_000),
    raw("vendor/expiring:free", inputs=("text", "image"), params=FULL, expiration_date="2026-12-31"),
    raw("vendor/forced-think:free", params=FULL, reasoning={"mandatory": True}, context_length=8000),
    raw("vendor/vision:free", inputs=("text", "image"), params=FULL, context_length=4000),
    raw("other/text:free", params=FULL, context_length=2000),
    raw("vendor/tools-only:free", params=("tools",), context_length=7000),
    raw("vendor/no-structure:free", params=("temperature",), context_length=7000),
]


def test_free_filter_needs_both_prices_zero_and_drops_routers_and_non_text_output():
    ids = {m.id for m in pfm.free_models(LIST)}
    assert "vendor/paid" not in ids and "vendor/half-free:free" not in ids
    assert "openrouter/free" not in ids  # random router: forbidden for experiments
    assert "vendor/music:free" not in ids  # audio output: not a chat model
    assert "vendor/vision:free" in ids and "stealth/no-suffix" in ids  # listed, but see the picks test


def test_bad_or_missing_price_is_not_free():
    bad = [raw("a/b:free", prompt="n/a"), {"id": "c/d:free", "architecture": {}, "pricing": {}}]
    assert pfm.free_models(bad) == []


def test_parse_reads_capabilities():
    m = pfm.parse_model(LIST[7])
    assert (m.image, m.structured_outputs, m.response_format, m.tools, m.temperature) == (True,) * 5
    assert pfm.parse_model(LIST[6]).mandatory_reasoning is True
    assert pfm.parse_model(LIST[5]).expires == "2026-12-31"


def test_role_picks():
    picks = pfm.pick_roles(pfm.free_models(LIST))
    assert picks["vision"] == "vendor/vision:free"  # needs image input; the expiring one is skipped
    assert picks["strong"] == "vendor/vision:free"  # full capability, no forced reasoning, largest context
    assert picks["fast"] == picks["strong"]
    assert picks["judge"] == "other/text:free"  # a different model from the answerer
    assert "stealth/no-suffix" not in picks.values()  # free-tier limits are documented for :free ids


def test_judge_falls_back_to_answerer_when_alone():
    only = pfm.free_models([raw("vendor/vision:free", inputs=("text", "image"), params=FULL)])
    assert pfm.pick_roles(only)["judge"] == "vendor/vision:free"


def test_tools_only_model_is_usable_when_nothing_better_exists():
    models = pfm.free_models([raw("vendor/tools-only:free", inputs=("text", "image"), params=("tools",))])
    assert pfm.pick_roles(models)["strong"] == "vendor/tools-only:free"


def test_no_usable_model_raises():
    with pytest.raises(pfm.NoModelError):
        pfm.pick_roles(pfm.free_models([raw("vendor/no-structure:free", params=("temperature",))]))
    no_image = pfm.free_models([raw("other/text:free", params=FULL)])
    with pytest.raises(pfm.NoModelError):
        pfm.pick_roles(no_image)


def test_env_lines_are_ready_to_paste():
    lines = pfm.env_lines({"fast": "a:free", "strong": "b:free", "vision": "c:free", "judge": "d:free"})
    assert lines == ["MODEL_FAST=a:free", "MODEL_STRONG=b:free", "MODEL_VISION=c:free", "MODEL_JUDGE=d:free"]
