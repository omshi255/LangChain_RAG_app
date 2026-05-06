import logging
from typing import List, Dict, Optional
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

# In-memory store: { session_id: [ {role, content, timestamp}, ... ] }
_conversation_store: Dict[str, List[dict]] = defaultdict(list)

MAX_HISTORY_PER_SESSION = 20  # keep last N turns to avoid token overflow


def get_history(session_id: str) -> List[dict]:
    """Return conversation history for a session."""
    return _conversation_store[session_id]


def add_turn(session_id: str, question: str, answer: str) -> None:
    """Append a user question + assistant answer to the session history."""
    history = _conversation_store[session_id]

    history.append({
        "role": "user",
        "content": question,
        "timestamp": datetime.utcnow().isoformat(),
    })
    history.append({
        "role": "assistant",
        "content": answer,
        "timestamp": datetime.utcnow().isoformat(),
    })

    # Trim to last MAX_HISTORY_PER_SESSION turns (each turn = 2 messages)
    max_messages = MAX_HISTORY_PER_SESSION * 2
    if len(history) > max_messages:
        _conversation_store[session_id] = history[-max_messages:]

    logger.debug(f"Session '{session_id}': {len(_conversation_store[session_id])} messages stored")


def clear_history(session_id: str) -> None:
    """Delete all history for a session."""
    if session_id in _conversation_store:
        del _conversation_store[session_id]
        logger.info(f"Session '{session_id}' cleared")


def list_sessions() -> List[str]:
    """Return all active session IDs."""
    return list(_conversation_store.keys())


def build_chat_messages(session_id: str, system_prompt: str) -> List[dict]:
    """
    Build the full messages list for the Groq API call:
      [system] + [user/assistant history...] (without timestamps, Groq doesn't want those)
    """
    messages = [{"role": "system", "content": system_prompt}]

    for msg in _conversation_store[session_id]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"],
        })

    return messages