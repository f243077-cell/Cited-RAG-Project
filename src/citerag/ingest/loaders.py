"""File validation and text loading. No framework objects here: pages are plain dataclasses."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from citerag.errors import FileTooLargeError, InvalidFileTypeError

FileKind = Literal["pdf", "text", "image"]

_KIND_BY_EXTENSION: dict[str, FileKind] = {
    ".pdf": "pdf",
    ".md": "text",
    ".markdown": "text",
    ".txt": "text",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
}
_MAGIC = {".pdf": b"%PDF", ".png": b"\x89PNG\r\n\x1a\n", ".jpg": b"\xff\xd8\xff", ".jpeg": b"\xff\xd8\xff"}
_MAX_NAME_CHARS = 200


@dataclass(frozen=True)
class LoadedPage:
    page: int | None  # 1-based PDF page; None for text files
    text: str


@dataclass(frozen=True)
class LoadedDoc:
    pages: list[LoadedPage]
    page_count: int | None


def compute_doc_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def sanitize_name(name: str) -> str:
    """Display name only: files are stored under hashed names, and no path from user input reaches the filesystem."""
    base = PurePosixPath(name.replace("\\", "/")).name
    clean = re.sub(r"[^\w. -]", "_", base).lstrip(". ")
    return clean[-_MAX_NAME_CHARS:] or "upload"


def extension(name: str) -> str:
    return PurePosixPath(name.replace("\\", "/")).suffix.lower()


def _decode_text(name: str, data: bytes) -> str:
    if b"\x00" in data:
        raise InvalidFileTypeError(f"{name}: binary data in a text file")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InvalidFileTypeError(f"{name}: not UTF-8 text") from exc
    if not text.strip():
        raise InvalidFileTypeError(f"{name}: file is empty")
    return text


def validate_file(name: str, data: bytes, max_mb: int) -> FileKind:
    """Allow-list on the extension, then the size limit, then the content (magic bytes, not the name)."""
    suffix = extension(name)
    kind = _KIND_BY_EXTENSION.get(suffix)
    if kind is None:
        raise InvalidFileTypeError(f"{name}: unsupported file type {suffix or '(none)'}")
    if len(data) > max_mb * 1024 * 1024:
        raise FileTooLargeError(f"{name}: larger than the {max_mb} MB limit")
    if not data:
        raise InvalidFileTypeError(f"{name}: file is empty")
    if kind == "text":
        _decode_text(name, data)
    elif not data.startswith(_MAGIC[suffix]):
        raise InvalidFileTypeError(f"{name}: content does not match the {suffix} file type")
    return kind


def _load_pdf(data: bytes) -> LoadedDoc:
    try:
        reader = PdfReader(io.BytesIO(data))
        texts = [(page.extract_text() or "").strip() for page in reader.pages]
    except (PyPdfError, ValueError, KeyError, IndexError, TypeError, OSError) as exc:
        raise InvalidFileTypeError(f"PDF could not be read: {exc}") from exc
    pages = [LoadedPage(number, text) for number, text in enumerate(texts, start=1) if text]
    if not pages:
        raise InvalidFileTypeError("PDF has no extractable text (scanned PDFs are not supported)")
    return LoadedDoc(pages=pages, page_count=len(texts))


def load_pages(kind: FileKind, data: bytes) -> LoadedDoc:
    if kind == "pdf":
        return _load_pdf(data)
    if kind == "text":
        return LoadedDoc(pages=[LoadedPage(None, _decode_text("text file", data))], page_count=None)
    return LoadedDoc(pages=[], page_count=None)  # standalone images go through the image path in M5
