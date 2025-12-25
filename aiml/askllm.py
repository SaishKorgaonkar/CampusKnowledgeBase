from typing import Any, Dict
from rag import Retriever


class QAService:
    def __init__(self, client: Any, retriever: Retriever, model_name: str = "gemini-3-flash-preview"):
        self.client = client
        self.retriever = retriever
        self.model_name = model_name
    
    def ask(self, question: str, course:str = None, semester: str = None) -> Dict:
        retrieved_chunks = self.retriever.retrieve(question, course=course, semester=semester)

        context = "\n\n".join(f"- {c['text']}" for c in retrieved_chunks)

        prompt = f"""
You are a campus assistant.
Answer using the context below if present.
If the context is insufficient, say you don't know.

Context:
---
{context}

Question:
---
{question}
"""

        response = self.client.models.generate_content(model=self.model_name, contents=prompt)
        answer_text = response.text

        return {
            "answer": answer_text, 
            "sources": retrieved_chunks,
        }