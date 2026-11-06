"""Step 11 (LIVE, 1 request): small generated PNG through the standard LangChain image block, structured reply."""

import base64
import struct
import zlib

from common import chat_client, count_request, load_env, result, stop_on_quota

from langchain_core.messages import HumanMessage

from citerag.llm.parse import parse_model, schema_response_format
from citerag.schemas import ImageAnalysis


def red_png(size: int = 64) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(b"\x00" + b"\xff\x00\x00" * size for _ in range(size))
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


env = load_env()
model = env["MODEL_VISION"]
message = HumanMessage(
    content=[
        {"type": "image", "base64": base64.b64encode(red_png()).decode(), "mime_type": "image/png"},
        {"type": "text", "text": "Describe this image. Say what colour it is."},
    ]
)
count_request()
try:
    raw = chat_client(env, model).bind(response_format=schema_response_format(ImageAnalysis)).invoke([message])
    parsed = parse_model(ImageAnalysis, raw.content)
    result("red" in parsed.description.lower(), f"{model}: {parsed}; usage={raw.usage_metadata}")
except Exception as exc:  # noqa: BLE001
    stop_on_quota(exc)
    result(False, f"{model}: {type(exc).__name__}: {str(exc)[:300]}")
