from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

TraceStatus = Literal["completed", "failed"]
TraceSortField = Literal[
    "timestamp",
    "query",
    "retriever_name",
    "embedding_model",
    "status",
    "retrieval_latency_ms",
    "total_latency_ms",
]
TraceSortOrder = Literal["asc", "desc"]


def utc_now() -> datetime:
    return datetime.now(UTC)


class TraceChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float


class Trace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=utc_now)
    request_payload: dict[str, Any] = Field(default_factory=dict)
    query: str = Field(..., min_length=1)
    retriever_name: str = Field(..., min_length=1)
    embedding_model: str = Field(..., min_length=1)
    retrieved_chunks: list[TraceChunk] = Field(default_factory=list)
    similarity_scores: list[float] = Field(default_factory=list)
    retrieval_latency_ms: float = Field(default=0.0, ge=0)
    generation_latency_ms: float = Field(default=0.0, ge=0)
    total_latency_ms: float = Field(default=0.0, ge=0)
    status: TraceStatus
    error_message: str | None = None


class TraceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    timestamp: datetime
    query: str
    retriever_name: str
    embedding_model: str
    status: TraceStatus
    retrieval_latency_ms: float
    total_latency_ms: float


class TraceListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[TraceSummary]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    sort_by: TraceSortField
    sort_order: TraceSortOrder


class CreateTraceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str | None = None
    timestamp: datetime | None = None
    request_payload: dict[str, Any] = Field(default_factory=dict)
    query: str = Field(..., min_length=1)
    retriever_name: str = Field(..., min_length=1)
    embedding_model: str = Field(..., min_length=1)
    retrieved_chunks: list[TraceChunk] = Field(default_factory=list)
    similarity_scores: list[float] = Field(default_factory=list)
    retrieval_latency_ms: float = Field(default=0.0, ge=0)
    generation_latency_ms: float = Field(default=0.0, ge=0)
    total_latency_ms: float = Field(default=0.0, ge=0)
    status: TraceStatus
    error_message: str | None = None

    def to_trace(self) -> Trace:
        payload = self.model_dump(exclude_none=True)
        return Trace(**payload)
