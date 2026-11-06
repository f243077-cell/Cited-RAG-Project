"""Shared helpers for the throwaway preflight spikes. Live spikes count every HTTP attempt in a local file."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
COUNTER = ROOT / "data" / "preflight_requests.json"
MAX_REQUESTS = 20


def result(ok: bool, evidence: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}: {evidence}")
    if not ok:
        raise SystemExit(1)


def load_env() -> dict[str, str]:
    """Read .env without printing it. Exits "skipped: no key" when the key is missing."""
    from dotenv import dotenv_values

    env = {k: v for k, v in dotenv_values(ROOT / ".env").items() if v}
    if not env.get("OPENROUTER_API_KEY"):
        print("SKIPPED: no key (.env with OPENROUTER_API_KEY is missing)")
        raise SystemExit(0)
    return env


def count_request(n: int = 1) -> int:
    """Add n to the running count; refuses to go past the preflight cap of 20 requests."""
    used = json.loads(COUNTER.read_text())["used"] if COUNTER.exists() else 0
    if used + n > MAX_REQUESTS:
        print(f"SKIPPED: quota (used {used} of {MAX_REQUESTS} preflight requests)")
        raise SystemExit(0)
    COUNTER.parent.mkdir(parents=True, exist_ok=True)
    COUNTER.write_text(json.dumps({"used": used + n}))
    print(f"requests used so far: {used + n}/{MAX_REQUESTS}")
    return used + n


def stop_on_quota(exc: BaseException) -> None:
    """429, 402 or a daily-limit error ends all live checks: never retry in a loop."""
    status = getattr(exc, "status_code", None)
    if isinstance(exc, ValueError) and exc.args and isinstance(exc.args[0], dict):
        status = exc.args[0].get("code")  # the error can arrive inside an HTTP 200 body
    if status in (402, 429):
        print(f"SKIPPED: quota (HTTP {status}); stop all live checks now")
        raise SystemExit(0)


def chat_client(env: dict[str, str], model: str, **overrides):
    """ChatOpenAI built the way the gateway will build it (ARCHITECTURE section 11, note 1). No SDK retries."""
    from langchain_openai import ChatOpenAI

    extra = {"provider": {"require_parameters": env.get("OPENROUTER_REQUIRE_PARAMETERS", "true") == "true"}}
    return ChatOpenAI(
        model=model,
        base_url=env.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=env["OPENROUTER_API_KEY"],
        max_retries=0,
        timeout=60,
        max_tokens=2000,
        default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "CiteRAG"},
        extra_body=extra,
        **overrides,
    )


def set_api_key_env(env: dict[str, str]) -> None:
    """CrewAI's OpenRouter provider reads the key from the process environment."""
    os.environ["OPENROUTER_API_KEY"] = env["OPENROUTER_API_KEY"]
