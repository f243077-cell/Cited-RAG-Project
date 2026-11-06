"""Test helpers: a deterministic local embedder (no model download) for LlamaIndex and the cache."""

from __future__ import annotations

import hashlib
import math
import re

from llama_index.core.embeddings import BaseEmbedding


def hash_vector(text: str, dim: int = 128) -> list[float]:
    v = [0.0] * dim
    for w in re.findall(r"\w+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class HashEmbedding(BaseEmbedding):
    """Bag-of-words hashing: similar wording gives similar vectors. Enough to test retrieval plumbing."""

    def _get_query_embedding(self, query: str) -> list[float]:
        return hash_vector(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return hash_vector(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return hash_vector(query)
