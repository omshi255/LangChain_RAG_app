import logging
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import upload, query
from routers import image
from routers import memory  # ← NEW: memory/session management

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

app = FastAPI(
    title="Production RAG System with Memory",
    description=(
        "Upload documents and query them using Retrieval-Augmented Generation (RAG).\n\n"
        "**Memory feature:** Pass a `session_id` in your `/query` requests to maintain "
        "multi-turn conversation context. The assistant will remember previous questions "
        "and answers within the same session."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(upload.router)
app.include_router(query.router)
app.include_router(image.router, prefix="/api", tags=["Image"])
app.include_router(memory.router)   # ← NEW


@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "message": "RAG System with Memory is running. Visit /docs for API reference.",
    }


@app.get("/health", tags=["Health"])
def detailed_health():
    from config import get_settings
    from services.memory import list_sessions
    s = get_settings()
    return {
        "status": "ok",
        "llm_model": s.llm_model,
        "embedding_model": s.embedding_model,
        "pinecone_index": s.pinecone_index_name,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "active_sessions": len(list_sessions()),  # ← NEW
    }