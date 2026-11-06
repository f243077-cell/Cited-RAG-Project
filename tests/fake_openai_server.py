"""A tiny OpenAI-compatible chat-completions server for tests (stand-in for OpenRouter). No network, no keys."""

from __future__ import annotations

import http.server
import json
import threading


class FakeOpenRouter:
    """Usage:
        with FakeOpenRouter() as srv:
            srv.queue_text('{"answer": "x"}'); llm = ChatOpenAI(model="m", base_url=srv.base_url, api_key="k")
    Queued replies are served in order; when the queue is empty the last default text is returned."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        self._queue: list[tuple] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _send(self, code: int, obj: dict, headers: dict | None = None) -> None:
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                outer.headers.append({k.lower(): v for k, v in self.headers.items()})
                kind, arg = outer._queue.pop(0) if outer._queue else ("text", "OK")
                usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
                if kind == "429":
                    return self._send(
                        429, {"error": {"code": 429, "message": "rate limited"}}, {"Retry-After": str(arg)}
                    )
                if kind == "error200":
                    return self._send(200, {"error": {"code": 502, "message": "Provider returned error"}})
                if kind == "tool":
                    msg = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": arg[0], "arguments": json.dumps(arg[1])},
                            }
                        ],
                    }
                    fin = "tool_calls"
                else:
                    msg, fin = {"role": "assistant", "content": arg}, "stop"
                self._send(
                    200,
                    {
                        "id": "gen-1",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "vendor/model:free",
                        "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
                        "usage": usage,
                    },
                )

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/api/v1"

    def queue_text(self, text: str) -> None:
        self._queue.append(("text", text))

    def queue_tool(self, name: str, args: dict) -> None:
        self._queue.append(("tool", (name, args)))

    def queue_429(self, retry_after_s: float = 0) -> None:
        self._queue.append(("429", retry_after_s))

    def queue_error_in_200(self) -> None:
        self._queue.append(("error200", None))

    def __enter__(self) -> FakeOpenRouter:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()
