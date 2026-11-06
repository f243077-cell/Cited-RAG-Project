"""Step 16 (LOCAL): SentenceTransformerRerank with the MiniLM cross-encoder on 5 passages; report latency."""

import time

from common import result

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

texts = [
    "The warranty covers manufacturing defects for two years.",
    "Zebrafish are a common model organism in genetics.",
    "Items can be returned within thirty days of delivery.",
    "Shipping takes three to five working days.",
    "Refund requests are handled by the support team.",
]
nodes = [NodeWithScore(node=TextNode(id_=str(i), text=t), score=0.0) for i, t in enumerate(texts)]
start = time.perf_counter()
rerank = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=2)
load_s = time.perf_counter() - start
query = QueryBundle("How long do I have to return an item?")
runs = []
for _ in range(4):
    start = time.perf_counter()
    out = rerank.postprocess_nodes(nodes, query_bundle=query)
    runs.append(time.perf_counter() - start)
print(f"load+download {load_s:.1f}s; rerank 5 passages: first {runs[0] * 1000:.0f} ms, then {min(runs[1:]) * 1000:.0f} ms")
result(out[0].node.node_id == "2", f"top passage: {out[0].node.get_content()!r}")
