from fastapi import APIRouter, HTTPException
import logging
from models import QueryRequest, QueryResponse
from services.rag import run_rag_pipeline

router = APIRouter(prefix="/query", tags=["RAG Query"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=QueryResponse, summary="Query documents using RAG")
async def query_documents(request: QueryRequest):
    """
    Ask a question against indexed documents.
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=422, detail="Question cannot be empty.")

        result = run_rag_pipeline(
            question=request.question,
            top_k=request.top_k or 5,
            document_id=request.document_id,
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")