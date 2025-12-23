import json
import faiss
import numpy as np
from typing import Any, List, Dict
from embedder import Embedder


class Ingestor:
    def __init__(self, embedder: Embedder, index_dir: str = "index", chunks_path: str = "data/chunks.json"):
        self.embedder = embedder
        self.index_dir = index_dir
        self.chunks_path = chunks_path

    def ingest(self, chunks: List[Dict]):
        vectors = []
        metadata = []

        for chunk in chunks:
            vec = self.embedder.embed_text(chunk["text"])
            vectors.append(vec)
            metadata.append(chunk)

        vectors = np.vstack(vectors)

        # Create FAISS index
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)

        # Save index + metadata
        faiss.write_index(index, f"{self.index_dir}/faiss.index")

        with open(self.chunks_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return True
