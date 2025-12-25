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
                types.Part.from_text(text="""academic test"""),
            ],
        )
        # self.initial_history = [
        #     Content(
        #         role="system",
        #         parts=[Part(text="You are an academic research assistant. Use context evidence and cite sources respectfully.")]
        #     ),
        #     Content(
        #         role="user",
        #         parts=[Part(text="Hello, I’d like you to help me with research tasks.")]
        #     ),
        #     Content(
        #         role="system",
        #         parts=[Part(text="Please be thorough, cite any sources, and explain uncertainty when unclear.")]
        #     ),
        #     Content(
        #         role="user",
        #         parts=[Part(text="Let’s get started.")]
        #     ),
        # ]
        
        self.history_content = [
            Content(parts=[Part(text="""explain in detail drift velocity""")], role='user'),
            Content(parts=[Part(text="""YAHOOOOO"""),], role='model')
        ]

        self.chat = self.client.chats.create(model=self.model_name, config=generate_content_config, history=self.history_content)
        
    
    def ask(self, question: str, course:str = None, semester: str = None) -> Dict:
        retrieved_chunks = self.retriever.retrieve(question, course=course, semester=semester)

        context = "\n\n".join(f"- {c['text']}" for c in retrieved_chunks)

        prompt = f"""You are a campus assistant.
Answer using the context below if present.
If the context is insufficient, say you don't know.

Context:
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

        return {
            "answer": answer_text, 
            "sources": retrieved_chunks,
        }
