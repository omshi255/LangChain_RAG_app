import logging
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import upload, query
from routers import image
# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

app = FastAPI(
    title="Production RAG System",
    description="Upload PDFs and query them using Retrieval-Augmented Generation (RAG)",
    version="1.0.0",
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

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "message": "RAG System is running. Visit /docs for API reference."}


@app.get("/health", tags=["Health"])
def detailed_health():
    from config import get_settings
    s = get_settings()
    return {
        "status": "ok",
        "llm_model": s.llm_model,
        "embedding_model": s.embedding_model,
        "pinecone_index": s.pinecone_index_name,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
    }