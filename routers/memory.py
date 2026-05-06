from fastapi import APIRouter, HTTPException
import logging
from models import SessionHistoryResponse, ClearSessionResponse, ActiveSessionsResponse, ChatMessage
from services.memory import get_history, clear_history, list_sessions

router = APIRouter(prefix="/memory", tags=["Memory / Sessions"])
logger = logging.getLogger(__name__)


@router.get(
    "/sessions",
    response_model=ActiveSessionsResponse,
    summary="List all active session IDs",
)
def get_active_sessions():
    """Returns all session IDs that currently have conversation history in memory."""
    sessions = list_sessions()
    return ActiveSessionsResponse(sessions=sessions, total=len(sessions))


@router.get(
    "/sessions/{session_id}",
    response_model=SessionHistoryResponse,
    summary="Get conversation history for a session",
)
def get_session_history(session_id: str):
    """
    Retrieve the full conversation history (user + assistant turns)
    for the given `session_id`.
    """
    history = get_history(session_id)
    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"No history found for session '{session_id}'."
        )

    messages = [
        ChatMessage(role=m["role"], content=m["content"], timestamp=m.get("timestamp"))
        for m in history
    ]
    return SessionHistoryResponse(
        session_id=session_id,
        message_count=len(messages),
        messages=messages,
    )


@router.delete(
    "/sessions/{session_id}",
    response_model=ClearSessionResponse,
    summary="Clear conversation history for a session",
)
def delete_session_history(session_id: str):
    """
    Permanently delete the conversation history for the given `session_id`.
    Use this to start a fresh conversation under the same session ID,
    or to free memory after a session ends.
    """
    history = get_history(session_id)
    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"No history found for session '{session_id}'."
        )
    clear_history(session_id)
    logger.info(f"Session '{session_id}' cleared via API")
    return ClearSessionResponse(
        session_id=session_id,
        message="Conversation history cleared successfully.",
    )