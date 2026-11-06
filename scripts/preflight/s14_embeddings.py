"""Step 14 (LOCAL): download bge-small-en-v1.5, check 384 dimensions, unit norm, identical-text cosine."""

import time

from common import result

import numpy as np
from llama_index.core.utils import get_cache_dir
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

MODEL = "BAAI/bge-small-en-v1.5"
start = time.perf_counter()
embed = HuggingFaceEmbedding(model_name=MODEL)
load_s = time.perf_counter() - start

a = np.array(embed.get_text_embedding("Refunds are processed within five working days."))
b = np.array(embed.get_text_embedding("Refunds are processed within five working days."))
c = np.array(embed.get_text_embedding("The cat sat on the mat."))
q = np.array(embed.get_query_embedding("Refunds are processed within five working days."))

print(f"query_instruction={embed.query_instruction!r} text_instruction={embed.text_instruction!r}")
print(f"cosine text/text identical={a @ b:.6f}; unrelated={a @ c:.4f}; text/query identical={a @ q:.6f}")
import pathlib
folder = pathlib.Path(get_cache_dir()) / "models--BAAI--bge-small-en-v1.5"
size_mb = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file() and not p.is_symlink()) / 1e6
print(f"load+download {load_s:.1f}s; cache {folder}; model size {size_mb:.0f} MB")
result(
    len(a) == 384 and abs(np.linalg.norm(a) - 1) < 1e-3 and a @ b > 0.999,
    f"dim={len(a)} norm={np.linalg.norm(a):.6f}",
)
