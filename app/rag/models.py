from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.storage.models import ChunkRecord, DocumentRecord


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    text: str
    content_type: str
    parser_version: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_index: int
    text: str
    token_count: int
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document: DocumentRecord
    chunks: list[ChunkRecord]
