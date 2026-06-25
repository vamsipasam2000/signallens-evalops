from __future__ import annotations

from copy import deepcopy
from typing import Protocol

from app.storage.models import ChunkRecord, DocumentRecord


class DocumentRepository(Protocol):
    def save_document(self, record: DocumentRecord) -> DocumentRecord:
        ...

    def save_chunks(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        ...

    def get_document(self, document_id: str) -> DocumentRecord | None:
        ...

    def list_chunks(self, document_id: str) -> list[ChunkRecord]:
        ...

    def list_documents(self) -> list[DocumentRecord]:
        ...

    def list_all_chunks(self) -> list[ChunkRecord]:
        ...


class InMemoryDocumentRepository:
    """Local development repository matching the PostgreSQL logical model."""

    def __init__(self) -> None:
        self._documents: dict[str, DocumentRecord] = {}
        self._chunks: dict[str, list[ChunkRecord]] = {}

    def save_document(self, record: DocumentRecord) -> DocumentRecord:
        self._documents[record.document_id] = deepcopy(record)
        self._chunks.setdefault(record.document_id, [])
        return deepcopy(record)

    def save_chunks(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        for chunk in chunks:
            self._chunks.setdefault(chunk.document_id, []).append(deepcopy(chunk))
        return deepcopy(chunks)

    def get_document(self, document_id: str) -> DocumentRecord | None:
        record = self._documents.get(document_id)
        return deepcopy(record) if record is not None else None

    def list_chunks(self, document_id: str) -> list[ChunkRecord]:
        return deepcopy(self._chunks.get(document_id, []))

    def list_documents(self) -> list[DocumentRecord]:
        return deepcopy(list(self._documents.values()))

    def list_all_chunks(self) -> list[ChunkRecord]:
        return deepcopy(
            [
                chunk
                for chunks in self._chunks.values()
                for chunk in chunks
            ]
        )

    def clear(self) -> None:
        self._documents.clear()
        self._chunks.clear()
