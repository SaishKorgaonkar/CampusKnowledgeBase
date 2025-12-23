import json
import faiss
import numpy as np
from typing import Any, List, Dict
from embedder import Embedder


class Retriever:
    def __init__(self, embedder: Embedder, index_path: str = "index/faiss.index", chunks_path: str = "data/chunks.json"):
        self.embedder = embedder
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.index = faiss.read_index(self.index_path)
        with open(self.chunks_path) as f:
            self.chunks = json.load(f)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        q_vec = self.embedder.embed_text(query).reshape(1, -1)
        distances, indices = self.index.search(q_vec, top_k)

        results: List[Dict] = []
        for idx in indices[0]:
            if idx == -1:
                continue
            results.append(self.chunks[idx])

        return results
