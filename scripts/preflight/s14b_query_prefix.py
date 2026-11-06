"""Step 14b (LOCAL): which constructor stops LlamaIndex prepending its bge query instruction?"""
from common import result
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

S = "Refunds are processed within five working days."
for label, kw in (("default", {}), ("empty string", {"query_instruction": "", "text_instruction": ""}), ("space", {"query_instruction": " ", "text_instruction": " "})):
    e = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", **kw)
    t, q = np.array(e.get_text_embedding(S)), np.array(e.get_query_embedding(S))
    print(f"{label}: text/query cosine={t @ q:.6f}; prompts={e._model.prompts}")
result(True, "see lines above")
