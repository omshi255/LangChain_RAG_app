import logging
from typing import List, Optional
from groq import Groq
from config import get_settings
from services.embeddings import embed_query
from services.vector_store import similarity_search
from models import QueryResponse, SourceChunk

logger = logging.getLogger(__name__)
settings = get_settings()

_groq_client = None


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


SYSTEM_PROMPT = """You are a helpful document assistant. You are given extracted text from documents or images (via OCR).
Your job is to describe, summarize, or answer questions based on the provided context.
Rules:
- If asked "what is the content" or "what does this contain", summarize everything in the context clearly.
- If asked a specific question, answer it from the context.
- If context is truly empty or unrelated, say: "I don't have enough information in the provided documents to answer this."
- Never fabricate facts outside the context."""


def build_context_block(chunks: List[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        page_info = f" (Page {chunk['page']})" if chunk.get("page") else ""
        parts.append(f"[Chunk {i}{page_info}]:\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, context: str) -> str:
    client = get_groq_client()

    # If user is asking about content/summary, reframe the prompt
    summary_triggers = [
        "what is content",
        "what is the content",
        "content of image",
        "what does it contain",
        "summarize",
        "what is in",
        "tell me about",
        "describe",
        "what information",
        "extract",
    ]
    is_summary_request = any(trigger in question.lower() for trigger in summary_triggers)

    if is_summary_request:
        user_message = f"""The following is text extracted from a document or image via OCR:

{context}

Please summarize and describe all the information present in this extracted text clearly and completely."""
    else:
        user_message = f"""Context from documents:
{context}

Question: {question}

Answer based strictly on the context above:"""

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def run_rag_pipeline(
    question: str,
    top_k: int = 5,
    document_id: Optional[str] = None,
) -> QueryResponse:
    logger.info(f"RAG query: '{question[:80]}'")

    query_embedding = embed_query(question)
    chunks = similarity_search(query_embedding, top_k=top_k, document_id=document_id)

    if not chunks:
        return QueryResponse(
            answer="I couldn't find any relevant information in the documents to answer your question.",
            sources=[],
            question=question,
        )

    context = build_context_block(chunks)
    answer = generate_answer(question, context)
    sources = [
        SourceChunk(text=c["text"], page=c.get("page"), score=c["score"])
        for c in chunks
    ]

    logger.info(f"RAG complete. Sources: {len(sources)}")
    return QueryResponse(answer=answer, sources=sources, question=question)