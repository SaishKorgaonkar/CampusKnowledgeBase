import os
from typing import Any, Optional

import numpy as np
from google import genai


DEFAULT_EMBED_MODEL = "gemini-embedding-001"


def create_gemini_client() -> Any:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment")
    return genai.Client(api_key=api_key)


class Embedder:
    def __init__(self, client: Optional[Any] = None, model: str = DEFAULT_EMBED_MODEL):
        self.client = client or create_gemini_client()
        self.model = model

    def embed_text(self, text: str) -> np.ndarray:
        response = self.client.models.embed_content(model=self.model, contents=text)
        # SDK response shapes vary by version:
        # - older: response.embedding -> list[float]
        # - newer: response.embeddings -> list[Embedding], each with `.values`
        if hasattr(response, "embedding"):
            values = response.embedding
        elif hasattr(response, "embeddings") and response.embeddings:
            first = response.embeddings[0]
            if hasattr(first, "values"):
                values = first.values
            elif hasattr(first, "embedding"):
                values = first.embedding
            else:
                raise AttributeError("Unexpected embedding object shape in response.embeddings[0]")
        else:
            raise AttributeError("EmbedContentResponse has no 'embedding' or 'embeddings' data")

        return np.array(values, dtype="float32")
