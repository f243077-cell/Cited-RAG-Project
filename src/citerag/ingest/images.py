"""Images inside documents (F-MM-3): extract, filter, and keep captions on disk so no image is paid for twice.

Sizes and PNG conversion go through pypdf's own image objects, so this file needs no direct Pillow import.
"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from citerag.schemas import ImageCaption

logger = logging.getLogger(__name__)
COVER_PAGES = 2  # covers and logos sit here: about a quarter of the sample images, at one LLM request each
MIN_PAGES_FOR_COVER = 5  # in a short PDF the first page may be the content


@dataclass(frozen=True)
class FoundImage:
    sha256: str
    data: bytes
    media_type: str
    page: int | None
    raw_size: int


@dataclass(frozen=True)
class ImageBatch:
    found: int
    kept: list[FoundImage]


def is_cover(page: int, page_count: int) -> bool:
    return page_count >= MIN_PAGES_FOR_COVER and page <= COVER_PAGES


def _png_bytes(image: object) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "PNG")
    return buffer.getvalue()


def pdf_images(data: bytes, min_px: int, min_bytes: int, cap: int) -> ImageBatch:
    """Skips icons, covers and duplicates, then keeps the largest images up to the cap."""
    reader = PdfReader(io.BytesIO(data))
    count = len(reader.pages)
    found = 0
    kept: dict[str, tuple[int, int, object]] = {}
    for number, page in enumerate(reader.pages, start=1):
        try:
            images = list(page.images)
        except (PyPdfError, OSError, ValueError, KeyError):
            logger.warning("page %s: could not list images", number)
            continue
        for image in images:
            found += 1
            sha = hashlib.sha256(image.data).hexdigest()
            if sha in kept or len(image.data) < min_bytes or is_cover(number, count):
                continue
            try:
                width, height = image.image.size
            except (OSError, ValueError, AttributeError):
                logger.warning("page %s: could not read an image", number)
                continue
            if width >= min_px and height >= min_px:
                kept[sha] = (number, len(image.data), image.image)
    largest = sorted(kept.items(), key=lambda pair: -pair[1][1])[:cap]
    return ImageBatch(
        found,
        [FoundImage(sha, _png_bytes(img), "image/png", page, raw) for sha, (page, raw, img) in largest],
    )


class ImageStore:
    """data/images/<sha256>.<suffix> and data/captions/<sha256>.json."""

    def __init__(self, data_dir: Path) -> None:
        self._images, self._captions = data_dir / "images", data_dir / "captions"

    def _caption_path(self, sha256: str) -> Path:
        return self._captions / f"{sha256}.json"

    def cached_caption(self, sha256: str) -> ImageCaption | None:
        path = self._caption_path(sha256)
        if not path.exists():
            return None
        try:
            return ImageCaption.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:  # a broken cache file costs one request, not a failed upload
            logger.warning("ignoring an unreadable caption file %s", path.name)
            return None

    def save_caption(self, sha256: str, caption: ImageCaption) -> None:
        self._captions.mkdir(parents=True, exist_ok=True)
        self._caption_path(sha256).write_text(caption.model_dump_json(), encoding="utf-8")

    def save_image(self, image: FoundImage) -> Path:
        self._images.mkdir(parents=True, exist_ok=True)
        path = self._images / f"{image.sha256}.{image.media_type.split('/')[1]}"
        if not path.exists():
            path.write_bytes(image.data)
        return path


def caption_text(caption: ImageCaption) -> str:
    """The indexed text: caption, visible text and key facts."""
    parts = [caption.caption, caption.visible_text, *caption.key_facts]
    return "\n".join(part for part in parts if part)
