"""Step 6 (LIVE, key-info call, no model request): tier and remaining free-model requests. Never prints the key."""

from common import load_env, result

import httpx

env = load_env()
base = env.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
r = httpx.get(f"{base}/key", headers={"Authorization": f"Bearer {env['OPENROUTER_API_KEY']}"}, timeout=30)
r.raise_for_status()
d = r.json()["data"]
safe = {k: d.get(k) for k in ("is_free_tier", "limit", "limit_remaining", "usage_daily", "free_model_daily_requests")}
result(True, f"key status: {safe}")
