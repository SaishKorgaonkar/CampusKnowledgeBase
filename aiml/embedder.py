import numpy as np
from typing import Any


class Embedder:
    def __init__(self, client: Any, model: str = "gemini-embedding-001"):
        self.client = client
        self.model = model

    def embed_text(self, text: str) -> np.ndarray:
        response = self.client.models.embed_content(model=self.model, contents=text)
        return np.array(response.embedding, dtype="float32")
