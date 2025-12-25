# 🎓 CampusKnowledgeBase

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Next.js](https://img.shields.io/badge/Next.js-13+-000000?logo=next.js&logoColor=white)](https://nextjs.org/)
[![Gemini API](https://img.shields.io/badge/Gemini-API-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-FFD700)](https://faiss.ai/)
[![Google OAuth](https://img.shields.io/badge/Google%20OAuth-Auth-4285F4?logo=google&logoColor=white)](https://developers.google.com/identity)

---

## 🎯 One-Liner

**An intelligent campus knowledge assistant that instantly answers student questions with cited references from course materials using RAG and Gemini AI.**

---

## 📸 Project Highlights

This is a **Retrieval-Augmented Generation (RAG)** system designed to help students navigate vast course materials efficiently:

- **Live Semester Structure**: Organizes FY and SY course materials by semester and subject
- **Accuracy Scoring**: Returns confidence scores for each answer based on source alignment
- **Secure Access**: Google OAuth integration for campus authentication
- **Real-time Retrieval**: FAISS vector search indexes over 10,000+ document chunks

---

## 🚨 The Problem

Students face information overload:
- 📚 Navigate hundreds of pages of PDFs across multiple subjects
- ⏱️ Spend hours searching for relevant course material
- 🤔 Can't verify if information is correct without manual checking
- 🔒 Sensitive campus materials need restricted access

**CampusKnowledgeBase solves this** by instantly retrieving relevant course content and synthesizing answers with citation accuracy.

---

## ✨ Key Features

- **📖 Contextual RAG Engine**: Uses Google Gemini 1.5 Flash to generate accurate, cited answers
- **⚡ Vector Search**: FAISS-powered semantic search for millisecond-level retrieval from 10,000+ chunks
- **🔐 Student Authentication**: Secure Google Sign-In restricted to campus members only
- **📊 Accuracy Scoring**: Evaluates answer confidence based on retrieved source alignment (0-1 scale)
- **🎓 Semester-Aware Filtering**: Retrieves materials specific to FY Sem-1, SY Sem-3, etc.
- **🔗 Source Citations**: Returns top-3 referenced documents for answer verification
- **💬 Multi-Subject Support**: Covers core subjects (DSA, COA, Physics, Maths, Chemistry, Biology, etc.)

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend API** | Flask + Python | REST API, RAG orchestration |
| **AI/ML** | Google Gemini API | LLM for answer generation & evaluation |
| **Embeddings** | Google Text Embeddings | Semantic text representation |
| **Vector Database** | FAISS | Sub-millisecond similarity search |
| **Frontend** | Next.js + TypeScript | Interactive chat UI |
| **Authentication** | Google OAuth 2.0 | Secure student sign-in |
| **Infrastructure** | Flask Dev Server | Can be deployed on Cloud Run |
| **Data** | Campus course PDFs | Processed into chunks + embeddings |

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:

- **Python**: 3.9 or higher
- **Node.js**: 16+ (for frontend)
- **pip**: Python package manager
- **npm/yarn**: Node package manager
- **Git**: Version control

### Environment Setup

Create a `.env` file in the root directory with the following variables:

```env
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Flask Configuration
FLASK_SECRET_KEY=your_secret_key_here
FLASK_ENV=development

# Frontend URL (for CORS)
FRONTEND_URL=http://localhost:3000

# Google OAuth (optional, for auth)
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
```

### Installation Steps

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/CampusKnowledgeBase.git
cd CampusKnowledgeBase
```

#### 2️⃣ Backend Setup (Python + Flask)

```bash
# Navigate to backend
cd aiml

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Flask server
python main.py
```

Server runs at: `http://localhost:8000`

#### 3️⃣ Frontend Setup (Next.js)

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install  # or yarn install

# Start development server
npm run dev  # or yarn dev
```

Frontend runs at: `http://localhost:3000`

#### 4️⃣ (Optional) Ingest Course Materials

To add new course materials:

```bash
cd aiml/ingestion
python ingest.py --input /path/to/pdfs --output ./output
```

---

## 📁 Project Structure

```
CampusKnowledgeBase/
│
├── aiml/                                  # AI/ML Backend
│   ├── main.py                           # Flask application entry point
│   ├── askllm.py                         # QA Service with accuracy scoring
│   ├── rag.py                            # Retriever (FAISS + semantic search)
│   ├── embedder.py                       # Text embedding using Google API
│   ├── config.py                         # Flask configuration
│   ├── requirements.txt                  # Python dependencies
│   ├── auth/                             # Authentication
│   │   └── google_oauth.py               # Google OAuth setup
│   ├── routes/                           # API Routes
│   │   └── auth_routes.py                # Auth endpoints
│   └── ingestion/                        # Data Processing
│       ├── ingest.py                     # PDF ingestion pipeline
│       ├── chunker.py                    # Document chunking logic
│       └── output/
│           ├── chunks.jsonl              # Processed document chunks
│           ├── faiss.index               # Vector search index
│           └── progress.json             # Ingestion progress tracker
│
├── frontend/                              # Next.js Frontend
│   ├── src/
│   │   ├── app/                          # Next.js app directory
│   │   │   ├── page.tsx                  # Landing page
│   │   │   ├── layout.tsx                # Root layout
│   │   │   ├── chat/                     # Chat interface
│   │   │   ├── login/                    # Login page
│   │   │   └── auth/                     # Auth pages
│   │   ├── components/                   # Reusable React components
│   │   │   ├── Button.tsx
│   │   │   ├── Card.tsx
│   │   │   ├── Input.tsx
│   │   │   ├── Textarea.tsx
│   │   │   ├── Navbar.tsx
│   │   │   ├── Footer.tsx
│   │   │   └── chat/                     # Chat-specific components
│   │   ├── contexts/                     # React contexts
│   │   │   ├── SemesterContext.tsx       # Semester selection state
│   │   │   └── ThemeContext.tsx          # Theme management
│   │   └── types/                        # TypeScript type definitions
│   │       └── chat.ts                   # Chat types
│   ├── package.json
│   ├── tsconfig.json
│   └── next.config.ts
│
├── data/                                  # Course Materials (organized by year/semester)
│   ├── FY/                               # First Year
│   │   └── Sem-1/
│   │       ├── Bee/                      # Basic Electrical Engineering
│   │       ├── Bio/                      # Biology
│   │       ├── Chem/                     # Chemistry
│   │       ├── Maths/                    # Mathematics
│   │       ├── Physics/                  # Physics
│   │       ├── SPM/                      # Structured Programming
│   │       └── [subjects].../
│   │
│   └── SY/                               # Second Year
│       └── SEM 3/
│           ├── DSA/                      # Data Structures & Algorithms
│           ├── COA/                      # Computer Organization & Architecture
│           ├── DDL/                      # Database Design & Languages
│           ├── OOPM/                     # Object-Oriented Programming
│           └── [subjects].../
│
└── README.md                              # This file
```

---

## 🔧 API Endpoints

### `/ask` (POST)
Ask a question and retrieve an answer with sources.

**Request:**
```json
{
  "question": "What is a binary search tree?",
  "semester": "FY-Sem-1",
  "course": "FY"
}
```

**Response:**
```json
{
  "answer": "A binary search tree is a data structure where each node has at most two children...",
  "sources": [
    {
      "text": "BST definition from Module 3...",
      "course": "DSA",
      "semester": "2"
    },
    ...
  ],
  "accuracy_score": 0.87
}
```

### `/auth-test` (GET)
Test OAuth authentication.

**Response:**
```json
{
  "message": "OAuth is working 🎉",
  "email": "student@somaiya.edu",
  "role": "student"
}
```

---

## 📊 How Accuracy Scoring Works

The system evaluates answer quality on a **0-1 scale**:

1. **Question** + **Generated Answer** + **Retrieved Context** are sent to Gemini
2. Model evaluates: *"How well is this answer supported by the given sources?"*
3. Score returned: `0.0` (not supported) to `1.0` (perfectly supported)
4. **Fallback Heuristic**: Word overlap between context and answer if API call fails

---

## 🙏 Acknowledgments

- Google Gemini API for powerful LLM capabilities
- FAISS by Facebook/Meta for vector search
- Campus knowledge base contributors
- The student community for feedback and insights

---

## 📧 Contact & Support

Have questions or found a bug? Open an [issue](https://github.com/your-username/CampusKnowledgeBase/issues) or reach out to us!

**Built with ❤️ by Saish, Shaurya, Soha and Bhoumik.**