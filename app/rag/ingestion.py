from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

from app.core.errors import IngestionValidationError
from app.rag.chunking import TokenWindowChunker
from app.rag.embeddings import EmbeddingProvider
from app.rag.models import IngestionResult
from app.rag.parsers import DocumentParserRouter
from app.retrieval.retrievers import VectorIndex
from app.storage.models import ChunkRecord, DocumentRecord
from app.storage.repositories import DocumentRepository

ChunkerFactory = Callable[[int, int], TokenWindowChunker]


class IngestionService:
    def __init__(
        self,
        *,
        parser_router: DocumentParserRouter,
        chunker_factory: ChunkerFactory,
        embedding_provider: EmbeddingProvider,
        repository: DocumentRepository,
        default_chunk_size_tokens: int,
        default_chunk_overlap_tokens: int,
        vector_index: VectorIndex | None = None,
    ) -> None:
        self._parser_router = parser_router
        self._chunker_factory = chunker_factory
        self._embedding_provider = embedding_provider
        self._repository = repository
        self._default_chunk_size_tokens = default_chunk_size_tokens
        self._default_chunk_overlap_tokens = default_chunk_overlap_tokens
        self._vector_index = vector_index

    def ingest_bytes(
        self,
        *,
        document_name: str,
        content: bytes,
        content_type: str | None,
        source: str = "api",
        metadata: dict[str, Any] | None = None,
        chunk_size_tokens: int | None = None,
        chunk_overlap_tokens: int | None = None,
    ) -> IngestionResult:
        if not document_name.strip():
            raise IngestionValidationError("document_name is required")
        if not content:
            raise IngestionValidationError("document content is empty")

        started_at = perf_counter()
        parsed = self._parser_router.parse(
            filename=document_name,
            content=content,
            content_type=content_type,
        )
        document_metadata = {
            **(metadata or {}),
            **parsed.metadata,
        }
        chunker = self._chunker_factory(
            chunk_size_tokens or self._default_chunk_size_tokens,
            chunk_overlap_tokens
            if chunk_overlap_tokens is not None
            else self._default_chunk_overlap_tokens,
        )
        text_chunks = chunker.chunk(
            parsed.text,
            metadata={
                **document_metadata,
                "document_name": document_name,
                "content_type": parsed.content_type,
            },
        )
        embeddings = self._embedding_provider.embed_documents([chunk.text for chunk in text_chunks])
        token_count = len(parsed.text.split())
        ingestion_latency_ms = round((perf_counter() - started_at) * 1000, 3)

        document = self._repository.save_document(
            DocumentRecord(
                name=document_name,
                content_type=parsed.content_type,
                source=source,
                metadata=document_metadata,
                chunk_count=len(text_chunks),
                token_count=token_count,
                ingestion_latency_ms=ingestion_latency_ms,
                parser_version=parsed.parser_version,
                chunker_version=chunker.version,
                embedding_model=self._embedding_provider.model_name,
            )
        )

        chunk_records = [
            ChunkRecord(
                document_id=document.document_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                token_count=chunk.token_count,
                metadata=chunk.metadata,
                embedding=embedding,
                embedding_model=self._embedding_provider.model_name,
            )
            for chunk, embedding in zip(text_chunks, embeddings, strict=True)
        ]
        stored_chunks = self._repository.save_chunks(chunk_records)
        if self._vector_index is not None:
            self._vector_index.upsert_chunks(stored_chunks)

        return IngestionResult(document=document, chunks=stored_chunks)
