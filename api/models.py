from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field
from config import CONFIDENCE_THRESHOLD


class UploadResponse(BaseModel):
    filename: str
    collection: str
    chunk_count: int
    char_count: int
    total_pages: Optional[int] = None
    has_tables: bool = False
    file_hash: str
    upload_timestamp: str
    warnings: List[str] = []
    trace_id: Optional[str] = None


class DocumentInfo(BaseModel):
    filename: str
    source_type: str
    chunk_count: int
    upload_timestamp: str
    collection: str
    has_table: bool = False


class ChunkDetail(BaseModel):
    id: int
    text: str
    heading: str
    source_file: str
    source_type: str
    chunk_index: Optional[int] = None
    upload_timestamp: Optional[str] = None
    has_table: Optional[bool] = None


class CollectionInfo(BaseModel):
    name: str
    vector_count: Optional[int] = None
    points_count: Optional[int] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    collection: str = "documents"
    top_k: int = Field(default=8, ge=1, le=20)
    filters: Optional[dict] = None
    use_hybrid: bool = True
    confidence_threshold: float = Field(default=CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)  # reads from config.py
    # Week 2 upgrade: Pydantic-validated JSON output (structured_answer.py) instead of free-text
    # parsing. Opt-in and defaults to False so the existing UI/clients are unaffected.
    use_structured_output: bool = False


class RetrievedChunk(BaseModel):
    text: str
    heading: str
    source_file: str
    source_type: str
    similarity: float
    rrf_score: Optional[float] = None
    chunk_index: Optional[int] = None


class QueryResponse(BaseModel):
    question: str
    answer: Optional[str] = None
    reason: Optional[str] = None
    message: Optional[str] = None
    confidence: float
    chunks: List[RetrievedChunk]
    collection: str
    model: str
    trace_id: Optional[str] = None
    trace_url: Optional[str] = None
    # Populated only when use_structured_output=True was requested. `answer` above is still set
    # (from structured.answer) so existing clients that only read `answer` keep working unchanged.
    quotes: Optional[List[str]] = None
    reasoning: Optional[str] = None
    sources: Optional[List[str]] = None
    is_refusal: Optional[bool] = None
