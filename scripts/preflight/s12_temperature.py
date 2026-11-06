"""Step 12 (LIVE, 1 request): does the judge model accept temperature=0?"""

from common import chat_client, count_request, load_env, result, stop_on_quota

env = load_env()
model = env["MODEL_JUDGE"]
count_request()
try:
    raw = chat_client(env, model, temperature=0).invoke("Reply with the single word OK.")
    result(True, f"{model} accepted temperature=0: {raw.content[:40]!r}")
except Exception as exc:  # noqa: BLE001
    stop_on_quota(exc)
    result(False, f"{model} rejected temperature=0: {type(exc).__name__}: {str(exc)[:300]}")
