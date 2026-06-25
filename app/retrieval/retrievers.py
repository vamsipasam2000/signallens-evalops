from __future__ import annotations

from time import perf_counter
from typing import Protocol

from app.rag.embeddings import EmbeddingProvider
from app.retrieval.filters import metadata_matches
from app.retrieval.models import RetrievalQuery, RetrievalResult, RetrievedChunk
from app.retrieval.similarity import cosine_similarity, keyword_overlap_score
from app.storage.models import ChunkRecord
from app.storage.repositories import DocumentRepository


class VectorIndex(Protocol):
    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        ...


class Retriever(Protocol):
    backend_name: str

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        ...


class RepositoryRetriever:
    backend_name = "memory"

    def __init__(
        self,
        *,
        repository: DocumentRepository,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        if not query.query.strip():
            raise ValueError("query must not be blank")
        if query.top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        started_at = perf_counter()
        query_embedding = self._embedding_provider.embed_documents([query.query])[0]
        scored_chunks: list[RetrievedChunk] = []

        for chunk in self._repository.list_all_chunks():
            if not metadata_matches(chunk.metadata, query.metadata_filter):
                continue
            score = cosine_similarity(query_embedding, chunk.embedding)
            if query.strategy == "keyword_boosted":
                score += 0.15 * keyword_overlap_score(query.query, chunk.text)

            scored_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    metadata=chunk.metadata,
                    embedding_model=chunk.embedding_model,
                    score=round(score, 6),
                )
            )

        ranked = sorted(scored_chunks, key=lambda item: item.score, reverse=True)[: query.top_k]
        latency_ms = round((perf_counter() - started_at) * 1000, 3)
        return RetrievalResult(
            query=query.query,
            top_k=query.top_k,
            strategy=query.strategy,
            backend=self.backend_name,
            embedding_model=self._embedding_provider.model_name,
            latency_ms=latency_ms,
            results=ranked,
        )
