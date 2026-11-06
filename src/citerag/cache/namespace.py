"""Fingerprints that decide which cached answers are still valid."""

from __future__ import annotations

import hashlib
import json


def _h(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def corpus_version(doc_ids: list[str], index_settings: dict) -> str:
    """Changes when a document is added, replaced or removed, or when chunking/embedding settings change."""
    return _h({"docs": sorted(doc_ids), "index": index_settings})


def cache_namespace(corpus: str, answer_model: str, answer_prompt_version: str, retrieval: dict) -> str:
    """Changes when ANYTHING that could change an answer changes. Cache entries from other namespaces never hit."""
    return _h(
        {"corpus": corpus, "model": answer_model, "prompt": answer_prompt_version, "retrieval": retrieval}
    )
