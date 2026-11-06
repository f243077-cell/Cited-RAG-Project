"""Vision calls (ARCHITECTURE section 10): the image a user attaches, and images found inside documents."""

from __future__ import annotations

import asyncio
import base64
import binascii
from collections.abc import Callable

from citerag.errors import ValidationFailedError
from citerag.llm.gateway import LLMGateway, StructuredResult
from citerag.prompts import load_prompt
from citerag.schemas import ImageAnalysis, ImageCaption

MAX_IMAGE_BYTES = 5 * 1024 * 1024
CAPTION_TIMEOUT_S = 60  # the whole caption, retries included: a stuck caption must not hold an upload
Captioner = Callable[[bytes, str], StructuredResult[ImageCaption]]


def _matches(media_type: str, data: bytes) -> bool:
    if media_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if media_type == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    return data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def validate_image(image_b64: str, media_type: str) -> None:
    """Never logs the bytes. The media type is checked against the content, not trusted from the client."""
    try:
        data = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValidationFailedError("The image is not valid base64") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise ValidationFailedError("The image is larger than 5 MB")
    if not _matches(media_type, data):
        raise ValidationFailedError(f"The image is not a valid {media_type}")


def image_block(image_b64: str, media_type: str) -> dict:
    """LangChain's standard block; it reaches the provider as an image_url part with a data URL."""
    return {"type": "image", "base64": image_b64, "mime_type": media_type}


async def analyze_image(
    llm: LLMGateway, image_b64: str, media_type: str, request_id: str
) -> StructuredResult[ImageAnalysis]:
    return await llm.structured_call(
        role="vision",
        schema=ImageAnalysis,
        system=load_prompt("vision").text,
        user=[image_block(image_b64, media_type), {"type": "text", "text": "Analyse this image."}],
        call_name="vision",
        request_id=request_id,
    )


async def caption_image(llm: LLMGateway, data: bytes, media_type: str) -> StructuredResult[ImageCaption]:
    image_b64 = base64.b64encode(data).decode("ascii")
    return await llm.structured_call(
        role="vision",
        schema=ImageCaption,
        system=load_prompt("caption").text,
        user=[image_block(image_b64, media_type), {"type": "text", "text": "Caption this image."}],
        call_name="caption",
        timeout_s=CAPTION_TIMEOUT_S,
    )


def blocking_captioner(llm: LLMGateway) -> tuple[Captioner, Callable[[], None]]:
    """For the CLI: one loop for the whole command, because the LLM client's connections belong to a loop."""
    loop = asyncio.new_event_loop()

    def caption(data: bytes, media_type: str) -> StructuredResult[ImageCaption]:
        return loop.run_until_complete(caption_image(llm, data, media_type))

    return caption, loop.close
