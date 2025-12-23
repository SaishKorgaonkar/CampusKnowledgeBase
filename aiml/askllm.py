from typing import Any, Dict
from rag import Retriever


class QAService:
    def __init__(self, client: Any, retriever: Retriever, model_name: str = "gemini-3.0-flash-preview"):
        self.client = client
        self.retriever = retriever
        self.model_name = model_name

    def ask(self, question: str) -> Dict:
        retrieved_chunks = self.retriever.retrieve(question)

        context = "\n\n".join(f"- {c['text']}" for c in retrieved_chunks)

        prompt = f"""
You are a campus assistant.
Answer ONLY using the context below.
If the answer is not present, say "I don’t know."

Context:
---
{context}

Question:
---
{question}
"""

        response = self.client.models.generate_content(model=self.model_name, contents=prompt)

        return {"answer": response.text, "sources": retrieved_chunks}