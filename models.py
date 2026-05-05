from pydantic import BaseModel, Field
from typing import Optional, List


class UploadResponse(BaseModel):
    message: str
    document_id: str
    chunks_stored: int
    filename: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    document_id: Optional[str] = None  # filter by doc if provided
    top_k: Optional[int] = Field(default=5, ge=1, le=20)


class SourceChunk(BaseModel):
    text: str
    page: Optional[int] = None
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    question: str