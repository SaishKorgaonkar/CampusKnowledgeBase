from flask import Flask, request, jsonify
from google import genai
import os
from dotenv import load_dotenv
from embedder import Embedder
from rag import Retriever
from askllm import QAService


load_dotenv()

app = Flask(__name__)

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Instantiate services
embedder = Embedder(client)
retriever = Retriever(embedder)
qa_service = QAService(client, retriever)


@app.route("/ask", methods=["POST"])
def ask_route():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    result = qa_service.ask(question)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
