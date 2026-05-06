# from pydantic import BaseModel, Field
# from typing import Optional, List


# class UploadResponse(BaseModel):
#     message: str
#     document_id: str
#     chunks_stored: int
#     filename: str


# class QueryRequest(BaseModel):
#     question: str = Field(..., min_length=3, max_length=1000)
#     document_id: Optional[str] = None  # filter by doc if provided
#     top_k: Optional[int] = Field(default=5, ge=1, le=20)


# class SourceChunk(BaseModel):
#     text: str
#     page: Optional[int] = None
#     score: float


# class QueryResponse(BaseModel):
#     answer: str
#     sources: List[SourceChunk]
#     question: str

from pydantic import BaseModel, Field
from typing import Optional, List


class UploadResponse(BaseModel):
    message: str
    document_id: str
    chunks_stored: int
    filename: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    document_id: Optional[str] = None       # filter by doc if provided
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    session_id: Optional[str] = Field(
        default=None,
        description="Conversation session ID. Pass the same ID across turns to enable memory."
    )


class SourceChunk(BaseModel):
    text: str
    page: Optional[int] = None
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    question: str
    session_id: Optional[str] = None        # echoed back so client can persist it


# ── Memory / Session models ──────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str                               # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None


class SessionHistoryResponse(BaseModel):
    session_id: str
    message_count: int
    messages: List[ChatMessage]


class ClearSessionResponse(BaseModel):
    session_id: str
    message: str


class ActiveSessionsResponse(BaseModel):
    sessions: List[str]
    total: int