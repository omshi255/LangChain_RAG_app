from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import logging
from models import QueryRequest, QueryResponse
from services.rag import run_rag_pipeline, run_rag_pipeline_stream

router = APIRouter(prefix="/query", tags=["RAG Query"])
logger = logging.getLogger(__name__)


@router.post(
    "/",
    response_model=QueryResponse,
    summary="Query documents using RAG (with memory)",
)
async def query_documents(request: QueryRequest):
    """
    Standard non-streaming query. Returns full answer once ready.

    - Pass `session_id` to enable multi-turn memory.
    - Omit `session_id` for a stateless one-off query.
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=422, detail="Question cannot be empty.")

        result = run_rag_pipeline(
            question=request.question,
            top_k=request.top_k or 5,
            document_id=request.document_id,
            session_id=request.session_id,
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post(
    "/stream",
    summary="Query documents using RAG with streaming (ChatGPT-like word-by-word)",
)
async def query_documents_stream(request: QueryRequest):
    """
    Streaming query — answer arrives token by token (Server-Sent Events).

    **How to read the stream:**
    - Each line starting with `data: ` is a token — append to your UI.
    - `data: [SOURCES] [...]` — JSON array of source chunks (at the end).
    - `data: [SESSION_ID] <id>` — session ID to use in next request.
    - `data: [DONE]` — stream is complete.

    **Postman:** Set request type to GET/POST and enable "Send and wait for response".
    **Frontend:** Use `EventSource` or `fetch` with `ReadableStream`.
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=422, detail="Question cannot be empty.")

        generator = run_rag_pipeline_stream(
            question=request.question,
            top_k=request.top_k or 5,
            document_id=request.document_id,
            session_id=request.session_id,
        )

        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",       # disables nginx buffering
                "Connection": "keep-alive",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Stream query failed: {str(e)}")