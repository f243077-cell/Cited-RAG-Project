"""Step 10 (LIVE, about 2 requests): json_schema path, then the tool-calling path only if the first fails."""

from common import chat_client, count_request, load_env, result, stop_on_quota

from pydantic import BaseModel

from citerag.llm.errors_map import classify_error
from citerag.llm.parse import parse_model, schema_response_format


class Capital(BaseModel):
    city: str
    country: str


env = load_env()
model = env["MODEL_STRONG"]
llm = chat_client(env, model)
msg = "Return the capital of France as JSON with fields city and country."


def report(method: str, raw, parsed: Capital) -> None:
    meta = raw.response_metadata
    result(
        True,
        f"{method} works on {model}: {parsed}; usage_metadata={raw.usage_metadata}; "
        f"token_usage={meta.get('token_usage')}; response_metadata keys={sorted(meta)}; "
        f"model_name={meta.get('model_name')}; model_provider={meta.get('model_provider')}",
    )


count_request()
try:
    raw = llm.bind(response_format=schema_response_format(Capital)).invoke(msg)
    report("json_schema", raw, parse_model(Capital, raw.content))
except Exception as exc:  # noqa: BLE001
    stop_on_quota(exc)
    print(f"json_schema failed: {classify_error(exc).kind}: {str(exc)[:300]}")

count_request()
try:
    raw = llm.bind_tools([Capital], tool_choice="Capital").invoke(msg)
    report("tools", raw, Capital.model_validate(raw.tool_calls[0]["args"]))
except Exception as exc:  # noqa: BLE001
    stop_on_quota(exc)
    result(False, f"both methods failed on {model}: {classify_error(exc).kind}: {str(exc)[:300]}")
