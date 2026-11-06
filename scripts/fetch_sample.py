"""Download the demo corpus into data/sample/ and record URL, licence and sha256 in data/sample/SOURCES.md (OPEN-1).

    uv run python scripts/fetch_sample.py

data/ is git-ignored, so this script is how the corpus travels. Files already present are not downloaded again.
Image counts use the same filter as ingestion (F-MM-3), because every captioned image is one LLM request.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import httpx
import typer
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"
MIN_IMAGE_PX = 100
MIN_IMAGE_BYTES = 5000
MAX_IMAGES_PER_DOC = 10

NIST_LICENCE = (
    "US government work, not subject to copyright in the US; NIST grants a worldwide royalty-free right to reprint "
    "(https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications)"
)
NIST_BASE = "https://nvlpubs.nist.gov/nistpubs"
RAW = "https://raw.githubusercontent.com"
FASTAPI_COMMIT = "1ca09c13584ca45aa0921e5e40b31d49c5e3135a"
UV_COMMIT = "24688aaca1e7fd104437a0909d5db3afdc7dcdb0"


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    licence: str


SOURCES = [
    Source("NIST.AI.100-1.pdf", f"{NIST_BASE}/ai/NIST.AI.100-1.pdf", NIST_LICENCE),
    Source("NIST.CSWP.29.pdf", f"{NIST_BASE}/CSWP/NIST.CSWP.29.pdf", NIST_LICENCE),
    Source("NIST.SP.800-218.pdf", f"{NIST_BASE}/SpecialPublications/NIST.SP.800-218.pdf", NIST_LICENCE),
    Source("NIST.SP.800-61r3.pdf", f"{NIST_BASE}/SpecialPublications/NIST.SP.800-61r3.pdf", NIST_LICENCE),
    Source(
        "fastapi-first-steps.md",
        f"{RAW}/fastapi/fastapi/{FASTAPI_COMMIT}/docs/en/docs/tutorial/first-steps.md",
        f"MIT ({RAW}/fastapi/fastapi/{FASTAPI_COMMIT}/LICENSE)",
    ),
    Source(
        "uv-readme.md",
        f"{RAW}/astral-sh/uv/{UV_COMMIT}/README.md",
        f"MIT or Apache-2.0 ({RAW}/astral-sh/uv/{UV_COMMIT}/LICENSE-MIT)",
    ),
]


def download(source: Source) -> Path:
    path = SAMPLE_DIR / source.name
    if path.exists():
        return path
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        response = client.get(source.url)
        response.raise_for_status()
    path.write_bytes(response.content)
    return path


def pdf_counts(path: Path) -> tuple[int, int]:
    """Returns (pages, distinct images that pass the ingestion size filter)."""
    reader = PdfReader(str(path))
    seen: set[str] = set()
    for page in reader.pages:
        for image in page.images:
            width, height = image.image.size
            if len(image.data) >= MIN_IMAGE_BYTES and min(width, height) >= MIN_IMAGE_PX:
                seen.add(hashlib.sha256(image.data).hexdigest())
    return len(reader.pages), len(seen)


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        "# Sample corpus sources",
        "",
        "| File | Pages | Images | sha256 | Licence | URL |",
        "|---|---|---|---|---|---|",
    ]
    total_pages = total_images = 0
    for source in SOURCES:
        path = download(source)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        pages, images = pdf_counts(path) if path.suffix == ".pdf" else (0, 0)
        total_pages += pages
        total_images += min(images, MAX_IMAGES_PER_DOC)
        rows.append(
            f"| {source.name} | {pages or '-'} | {images} | {digest} | {source.licence} | {source.url} |"
        )
        typer.echo(f"{source.name}: {pages} pages, {images} images, {path.stat().st_size} bytes")
    rows += ["", "Republished courtesy of the National Institute of Standards and Technology (NIST files)."]
    (SAMPLE_DIR / "SOURCES.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    typer.echo(f"total: {total_pages} PDF pages; at most {total_images} caption requests")


if __name__ == "__main__":
    main()
