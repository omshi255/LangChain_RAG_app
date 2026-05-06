import logging
import uuid
from typing import List, Optional, Generator
from groq import Groq
from config import get_settings
from services.embeddings import embed_query
from services.vector_store import similarity_search
from services.memory import get_history, add_turn, build_chat_messages
from models import QueryResponse, SourceChunk

logger = logging.getLogger(__name__)
settings = get_settings()

_groq_client = None


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


SYSTEM_PROMPT = """You are a helpful document assistant with memory of the current conversation.
You are given extracted text from documents or images (via OCR) as context.

Your job is to describe, summarize, or answer questions based on the provided context AND the conversation history.

Rules:
- Use the conversation history to understand follow-up questions and resolve pronouns (e.g. "it", "that", "the same").
- If asked "what is the content" or "what does this contain", summarize everything in the context clearly.
- If asked a specific question, answer it from the context.
- If context is truly empty or unrelated, say: "I don't have enough information in the provided documents to answer this."
- Never fabricate facts outside the context.
- If the user refers to something from earlier in the conversation (e.g. "summarize what we discussed"), use the conversation history.
"""


def build_context_block(chunks: List[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        page_info = f" (Page {chunk['page']})" if chunk.get("page") else ""
        parts.append(f"[Chunk {i}{page_info}]:\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def _build_messages(question: str, context: str, session_id: str) -> list:
    """Build Groq messages list with system prompt + history + current question."""
    messages = build_chat_messages(session_id, SYSTEM_PROMPT)

    summary_triggers = [
        "what is content", "what is the content", "content of image",
        "what does it contain", "summarize", "what is in",
        "tell me about", "describe", "what information", "extract",
    ]
    is_summary_request = any(trigger in question.lower() for trigger in summary_triggers)

    if is_summary_request:
        user_message = (
            f"The following is text extracted from a document or image via OCR:\n\n"
            f"{context}\n\n"
            f"Please summarize and describe all the information present in this extracted text clearly and completely."
        )
    else:
        user_message = (
            f"Context from documents:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer based strictly on the context above:"
        )

    messages.append({"role": "user", "content": user_message})
    return messages


# ─────────────────────────────────────────
# NON-STREAMING (existing behavior)
# ─────────────────────────────────────────

def generate_answer_with_memory(question: str, context: str, session_id: str) -> str:
    client = get_groq_client()
    messages = _build_messages(question, context, session_id)

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────
# STREAMING  ← NEW
# ─────────────────────────────────────────

def stream_answer_with_memory(
    question: str,
    context: str,
    session_id: str,
) -> Generator[str, None, None]:
    """
    Yields answer tokens one by one as they arrive from Groq.
    Caller is responsible for calling add_turn() after full answer is collected.
    """
    client = get_groq_client()
    messages = _build_messages(question, context, session_id)

    stream = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
        stream=True,          # ← only change vs non-streaming
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


# ─────────────────────────────────────────
# PIPELINES
# ─────────────────────────────────────────

def run_rag_pipeline(
    question: str,
    top_k: int = 5,
    document_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> QueryResponse:
    """Standard non-streaming pipeline — unchanged behavior."""
    logger.info(f"RAG query: '{question[:80]}' | session: {session_id}")

    if not session_id:
        session_id = str(uuid.uuid4())

    query_embedding = embed_query(question)
    chunks = similarity_search(query_embedding, top_k=top_k, document_id=document_id)

    if not chunks:
        answer = "I couldn't find any relevant information in the documents to answer your question."
        add_turn(session_id, question, answer)
        return QueryResponse(answer=answer, sources=[], question=question, session_id=session_id)

    context = build_context_block(chunks)
    answer = generate_answer_with_memory(question, context, session_id)
    add_turn(session_id, question, answer)

    sources = [
        SourceChunk(text=c["text"], page=c.get("page"), score=c["score"])
        for c in chunks
    ]
    logger.info(f"RAG complete. Sources: {len(sources)} | session: {session_id}")
    return QueryResponse(answer=answer, sources=sources, question=question, session_id=session_id)


def run_rag_pipeline_stream(
    question: str,
    top_k: int = 5,
    document_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """
    Streaming pipeline.
    Yields SSE-formatted strings:
      - data: <token>          — answer tokens
      - data: [SOURCES] ...    — sources JSON at the end
      - data: [DONE]           — signals stream end
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    logger.info(f"RAG stream query: '{question[:80]}' | session: {session_id}")

    query_embedding = embed_query(question)
    chunks = similarity_search(query_embedding, top_k=top_k, document_id=document_id)

    if not chunks:
        answer = "I couldn't find any relevant information in the documents to answer your question."
        add_turn(session_id, question, answer)
        yield f"data: {answer}\n\n"
        yield f"data: [SESSION_ID] {session_id}\n\n"
        yield "data: [DONE]\n\n"
        return

    context = build_context_block(chunks)

    # Stream tokens
    full_answer = ""
    for token in stream_answer_with_memory(question, context, session_id):
        full_answer += token
        # Escape newlines so SSE stays valid
        safe_token = token.replace("\n", "\\n")
        yield f"data: {safe_token}\n\n"

    # Save to memory after full answer collected
    add_turn(session_id, question, full_answer)

    # Send sources as a single SSE event at the end
    import json
    sources_data = [
        {"text": c["text"], "page": c.get("page"), "score": c["score"]}
        for c in chunks
    ]
    yield f"data: [SOURCES] {json.dumps(sources_data)}\n\n"
    yield f"data: [SESSION_ID] {session_id}\n\n"
    yield "data: [DONE]\n\n"

    logger.info(f"RAG stream complete. Sources: {len(chunks)} | session: {session_id}")