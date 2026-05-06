# 🤖 LangChain RAG App — Production Chatbot with Memory

A production-grade **Retrieval-Augmented Generation (RAG)** system built with **FastAPI**, **LangChain**, **Pinecone**, and **Groq**. Supports multi-turn conversation memory, streaming responses, query rewriting, and multi-format document uploads.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **Multi-format Upload** | PDF, DOCX, XLSX, CSV, PPTX, TXT, code files |
| 🖼️ **Image Upload** | OCR + Groq Vision fallback for image understanding |
| 🧠 **Conversation Memory** | Multi-turn chat with session-based history |
| 🌊 **Streaming Responses** | Word-by-word answer streaming (ChatGPT-like) |
| ✏️ **Query Rewriting** | Auto-rewrites vague follow-up questions using history |
| 🔍 **Semantic Search** | Pinecone vector DB with cosine similarity |
| ⚡ **Fast Embeddings** | FastEmbed (BAAI/bge-small-en) — runs locally, no API cost |
| 🦙 **Free LLM** | Groq `llama-3.1-8b-instant` — fast and free |

---

## 🗂️ Folder Structure

```
LangChain_RAG_app/
│
├── main.py                          # FastAPI app entry point, router registration
├── config.py                        # Settings & environment variables (Pydantic)
├── models.py                        # Request/Response Pydantic models
├── requirements.txt                 # Python dependencies
├── .gitignore
│
├── routers/                         # API route handlers
│   ├── __init__.py
│   ├── upload.py                    # POST /upload/ — any file upload & indexing
│   ├── query.py                     # POST /query/ and POST /query/stream
│   ├── image.py                     # POST /api/upload-image — image OCR + vision
│   └── memory.py                    # GET/DELETE /memory/sessions — session management
│
├── services/                        # Core business logic
│   ├── __init__.py
│   ├── rag.py                       # Main RAG pipeline (streaming + non-streaming)
│   ├── query_rewriter.py            # Rewrites vague queries using conversation history
│   ├── memory.py                    # In-memory session/conversation history store
│   ├── embeddings.py                # FastEmbed text → vector embeddings
│   ├── vector_store.py              # Pinecone upsert & similarity search
│   ├── ingestion.py                 # PDF text extraction & chunking (LangChain splitter)
│   └── ocr.py                       # Tesseract OCR for image text extraction
│
├── postman_collections/             # Ready-to-import Postman collections
│   ├── RAG_System.postman_collection.json   # Full system collection
│   ├── anydoc_upload.json           # Any file upload test
│   ├── pdf_upload.json              # PDF upload test
│   ├── image_upload.json            # Image upload test
│   ├── memory_sessions.json         # Memory & session management
│   └── streaming.json               # Streaming endpoint test
│
└── docs_to_upload/                  # Sample documents for testing
    ├── AI_ML_Guide.pptx
    ├── gestureIQ_report.pdf
    └── ...
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/omshi255/LangChain_RAG_app.git
cd LangChain_RAG_app
```

### 2. Create virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create `.env` file
```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=rag-index
PINECONE_ENVIRONMENT=us-east-1
```

> **Free API Keys:**
> - Groq → https://console.groq.com
> - Pinecone → https://app.pinecone.io

### 5. Run the server
```bash
uvicorn main:app --reload
```

Server starts at: `http://localhost:8000`
API Docs: `http://localhost:8000/docs`

---

## 🔌 API Endpoints

### 📤 Upload
| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload/` | Upload any file (PDF, DOCX, PPTX, CSV, XLSX, TXT, code) |
| POST | `/api/upload-image` | Upload image (OCR + Vision) |

### 💬 Query
| Method | Endpoint | Description |
|---|---|---|
| POST | `/query/` | Standard query — full answer at once |
| POST | `/query/stream` | Streaming query — word-by-word (SSE) |

### 🧠 Memory / Sessions
| Method | Endpoint | Description |
|---|---|---|
| GET | `/memory/sessions` | List all active sessions |
| GET | `/memory/sessions/{session_id}` | Get conversation history |
| DELETE | `/memory/sessions/{session_id}` | Clear session history |

### ❤️ Health
| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Basic health check |
| GET | `/health` | Detailed health + active sessions count |

---

## 🧪 How to Use (Postman)

### Step 1 — Upload a document
```json
POST /upload/
Body: form-data → file: your_document.pdf

Response:
{
  "document_id": "abc-123-xyz",
  "chunks_stored": 24,
  "filename": "your_document.pdf"
}
```
> ⚠️ **Save the `document_id`** — use it in all queries to filter results

---

### Step 2 — Query (Standard)
```json
POST /query/
{
  "question": "What is this document about?",
  "session_id": "my-session-001",
  "document_id": "abc-123-xyz",
  "top_k": 5
}

Response:
{
  "answer": "This document is about...",
  "question": "What is this document about?",
  "rewritten_question": null,
  "session_id": "my-session-001",
  "sources": [...]
}
```

---

### Step 3 — Follow-up (Query Rewriting in action)
```json
POST /query/
{
  "question": "tell me more about it",
  "session_id": "my-session-001",
  "document_id": "abc-123-xyz"
}

Response:
{
  "answer": "...",
  "rewritten_question": "What are the detailed aspects of the document topic?",
  "session_id": "my-session-001"
}
```

---

### Step 4 — Streaming Query
```json
POST /query/stream
Headers: Accept: text/event-stream
{
  "question": "Summarize the key points",
  "session_id": "my-session-001",
  "document_id": "abc-123-xyz",
  "top_k": 10
}

Stream response:
data: Based
data:  on
data:  the
data:  document...
data: [REWRITTEN_QUERY] Summarize the main key points of the document
data: [SOURCES] [{"text": "...", "score": 0.59}]
data: [SESSION_ID] my-session-001
data: [DONE]
```

---

## 🏗️ Architecture / Flow

```
User Query
    │
    ▼
Query Rewriter          ← rewrites vague follow-ups using session history
    │
    ▼
Embeddings (FastEmbed)  ← converts rewritten query to vector
    │
    ▼
Pinecone Search         ← finds top-K similar chunks
    │
    ▼
Context Builder         ← formats chunks into readable context block
    │
    ▼
Groq LLM (Llama 3.1)   ← generates answer using context + conversation history
    │
    ▼
Memory Store            ← saves Q&A turn to session history
    │
    ▼
Response → User         ← full JSON or SSE stream
```

---

## 🧠 Memory System

- Sessions stored **in-memory** (RAM) — resets on server restart
- Each session keeps last **20 turns** (40 messages) to avoid token overflow
- `session_id` is optional — omit it for stateless one-off queries
- Auto-generates a throwaway `session_id` if none provided

```python
# Turn 1
{ "question": "What is RAG?", "session_id": "chat-001" }

# Turn 2 — bot remembers Turn 1
{ "question": "Give me an example", "session_id": "chat-001" }

# View history
GET /memory/sessions/chat-001

# Start fresh
DELETE /memory/sessions/chat-001
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Web Framework | FastAPI |
| LLM | Groq — `llama-3.1-8b-instant` (free) |
| Vision Model | Groq — `meta-llama/llama-4-scout-17b-16e-instruct` |
| Embeddings | FastEmbed — `BAAI/bge-small-en` (local, free) |
| Vector DB | Pinecone (Serverless) |
| OCR | Tesseract via pytesseract |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Streaming | FastAPI StreamingResponse + Groq stream=True |
| Settings | Pydantic Settings + python-dotenv |

---

## 📦 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | — | Groq API key |
| `PINECONE_API_KEY` | ✅ Yes | — | Pinecone API key |
| `PINECONE_INDEX_NAME` | No | `rag-index` | Pinecone index name |
| `PINECONE_ENVIRONMENT` | No | `us-east-1` | Pinecone region |

---

## 🚀 Advanced RAG Features (Roadmap)

- [x] Conversation Memory
- [x] Streaming Responses
- [x] Query Rewriting
- [ ] Re-Ranking (Cross-Encoder)
- [ ] HyDE (Hypothetical Document Embeddings)
- [ ] Multi-Query Retrieval
- [ ] Contextual Compression
- [ ] Session Persistence (MongoDB/Redis)

---

## 👨‍💻 Author

**omshi255** — [GitHub](https://github.com/omshi255/LangChain_RAG_app)
