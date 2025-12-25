from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import os
from dotenv import load_dotenv

from embedder import Embedder
from rag import Retriever
from askllm import QAService

from config import Config
from extensions import jwt, oauth
from auth.google_oauth import init_google_oauth
from routes.auth_routes import auth_bp

from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY")
    
    # Enable CORS
    CORS(app, resources={
        r"/*": {
            "origins": os.getenv("FRONTEND_URL", "http://localhost:3000"),
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })

    jwt.init_app(app)
    init_google_oauth(app, oauth)

    app.register_blueprint(auth_bp)

    # Initialize Gemini client
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Instantiate services
    embedder = Embedder(client) # comment next 2 lines to test auth
    retriever = Retriever(embedder)
    qa_service = QAService(client, retriever)


    @app.route("/ask", methods=["POST"])
    @jwt_required()
    def ask_route():
        role = get_jwt()['role']

        if role!="student":
            return jsonify({"error": "Unauthorized"}), 403
        
        data = request.get_json()
        question = data.get("question", "")
        course = data.get("course", "FY") # Default to FY
        semester = data.get("semester", "Sem-1")  # Default to semester 1

        if not question:
            return jsonify({"error": "Question is required"}), 400

        result = qa_service.ask(question, course=course, semester=semester)
        return jsonify(result)
    
    @app.route("/auth-test", methods=["GET"])
    @jwt_required()
    def auth_test():
        email = get_jwt_identity()
        role = get_jwt()["role"]
        return {
            "message": "OAuth is working 🎉",
            "email": email,
            "role": role
        }
    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=8000)
