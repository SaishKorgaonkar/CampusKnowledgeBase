from typing import Dict
from rag import Retriever
from google.genai import Client
from google.api_core.exceptions import ResourceExhausted
from google.genai.types import Content, Part, GenerateContentConfig, UserContent
from google.genai import types

class QAService:
    def __init__(self, client: Client, retriever: Retriever, model_name: str = "gemini-2.5-flash"):
        self.client = client
        self.retriever = retriever
        self.model_name = model_name

        tools = [
            types.Tool(url_context=types.UrlContext()),
            types.Tool(google_search=types.GoogleSearch())
        ]
        generate_content_config = GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=5536,
            # thinking_config=ThinkingConfig(
            #     thinking_level="LOW",
            # ),
            tools=tools,
            system_instruction=[
                types.Part.from_text(text="""You are a helpful campus teaching assistant."""),
            ],
        )
        
        # self.history_content = [
        #     UserContent(parts=[Part(text="""explain in detail drift velocity""")]),
        #     Content(parts=[Part(text="""YAHOOOOO"""),], role='model'),
        #     UserContent(parts=[Part(text="""voltage explain""")]),
        #     Content(parts=[Part(text="""YAHOOOOO!!!!"""),], role='model'),
        #     UserContent(parts=[Part(text="""meow""")]),
        #     Content(parts=[Part(text="""YAHOOOOO!"""),], role='model'),
        # ]

        self.chat = self.client.chats.create(model=self.model_name, config=generate_content_config)#, history=self.history_content)
        
    
    def ask(self, question: str, course:str = None, semester: str = None) -> Dict:
        retrieved_chunks = self.retriever.retrieve(question, course=course, semester=semester)

        context = "\n\n".join(f"- {c['text']}" for c in retrieved_chunks)

        prompt = f"""
You are a helpful campus teaching assistant.

IMPORTANT RULES:
- Prefer answering using the information present in the context below.
- If the context does not contain enough information to answer, use google search tool to find the answer.
- If you still cannot find the answer, respond with:
  "I don't have enough information in the provided notes to answer this."

ANSWERING GUIDE:
1. Don't shorten the answer, keep it long and easy to understand for learner prupose.
2. Structure the answer with clear sections and headings (use KaTeX and Markdown).
3. Include examples, use bullet points, and step-by-step explanations where applicable.
4. Google search for extra information if context is insufficient.
5. Give the ORIGINAL context definition or statement in simple as well (do not oversimplify).
6. Use a SIMPLE ANALOGY or real-life comparison to make the idea intuitive.
7. If applicable, mention WHY this concept is important for exams or applications.
8. Keep the explanation clear, structured, and student-friendly.
   (do NOT give a single short summary paragraph unless asked).

TONE:
- Clear
- Student-friendly
- Exam-oriented
- No unnecessary jargon unless explained

RAG Search:
---
{context}

Question:
---
{question}
"""

        # response = self.client.models.generate_content(model=self.model_name, contents=prompt)
        response = self.chat.send_message(
            message=prompt
        )
        answer_text = response.text
        print(self.chat.get_history())

        # Format sources for frontend
        formatted_sources = []
        for idx, chunk in enumerate(retrieved_chunks):
            formatted_sources.append({
                "id": str(idx + 1),
                "fileName": chunk.get("doc_name", "Unknown"),
                "category": chunk.get("category", "Unknown"),
                "semester": chunk.get("semester", "Unknown"),
                "source": chunk.get("source", "Unknown"),
                "title": chunk.get("subject", "No Title"),
                "pageNumber": chunk.get("page", 0),
                "relevance": chunk.get("relevance", 0),  # You can calculate this based on similarity score
                "excerpt": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", ""),
                "filePath": chunk.get("source_path", "")
            })

        return {
            "answer": answer_text, 
            "sources": formatted_sources,
        }
