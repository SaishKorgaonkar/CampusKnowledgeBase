from typing import Any, Dict
import json
from rag import Retriever


class QAService:
    def __init__(self, client: Any, retriever: Retriever, model_name: str = "gemini-3-flash-preview"):
        self.client = client
        self.retriever = retriever
        self.model_name = model_name
    
    def _heuristic_accuracy_score(self, answer: str, retrieved_chunks: list) -> float:
        """
        heuristic scoring based on context relevance.
        """
        if not retrieved_chunks or not answer:
            return 0.0
        
        # Simple heuristic: count how many words from context appear in answer
        context_text = " ".join([c.get('text', '').lower() for c in retrieved_chunks])
        answer_lower = answer.lower()
        
        context_words = set(context_text.split())
        answer_words = set(answer_lower.split())
        
        if not answer_words:
            return 0.0
        
        overlap = len(context_words & answer_words)
        score = overlap / len(answer_words)
        return min(score, 1.0)

    def ask(self, question: str, course:str = None, semester: str = None) -> Dict:
        retrieved_chunks = self.retriever.retrieve(question, semester=semester)

        context = "\n\n".join(f"- {c['text']}" for c in retrieved_chunks)

        prompt = f"""
You are a campus assistant.
Answer using the context below if present.
Use google search preferbly for factual information if context is insufficient."

Context:
---
{context}

Question:
---
{question}
"""

        response = self.client.models.generate_content(model=self.model_name, contents=prompt)
        answer_text = response.text
        
        # Calculate accuracy score
        accuracy_score = self._calculate_accuracy_score(question, answer_text, retrieved_chunks)

        return {
            "answer": answer_text, 
            "sources": retrieved_chunks,
            "accuracy_score": accuracy_score
        }