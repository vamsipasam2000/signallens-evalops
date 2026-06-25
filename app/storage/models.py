from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

DocumentStatus = Literal["ingested", "failed"]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class DocumentRecord:
    name: str
    content_type: str
    source: str
    metadata: dict[str, Any]
    chunk_count: int
    token_count: int
    ingestion_latency_ms: float
    parser_version: str
    chunker_version: str
    embedding_model: str
    status: DocumentStatus = "ingested"
    error_message: str | None = None
    document_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class ChunkRecord:
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    metadata: dict[str, Any]
    embedding: list[float]
    embedding_model: str
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=utc_now)
