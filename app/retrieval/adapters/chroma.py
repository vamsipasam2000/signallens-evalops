from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from app.core.errors import DependencyUnavailableError
from app.rag.embeddings import EmbeddingProvider
from app.retrieval.filters import metadata_matches
from app.retrieval.models import RetrievalQuery, RetrievalResult, RetrievedChunk
from app.storage.models import ChunkRecord


class ChromaRetriever:
    backend_name = "chroma"

    def __init__(
        self,
        *,
        persist_directory: str,
        collection_name: str,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        try:
            import chromadb
        except ModuleNotFoundError as exc:
            raise DependencyUnavailableError(
                "ChromaDB is not installed. Install with `pip install -e '.[platform]'`."
            ) from exc

        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedding_provider = embedding_provider

    def ensure_collection(self) -> None:
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except ValueError:
            pass
        self.ensure_collection()

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            return

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[_chroma_metadata(chunk) for chunk in chunks],
        )

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        if not query.query.strip():
            raise ValueError("query must not be blank")
        if query.top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        started_at = perf_counter()
        query_embedding = self._embedding_provider.embed_documents([query.query])[0]
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=query.top_k,
            where=_chroma_where(query.metadata_filter),
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        chunks: list[RetrievedChunk] = []

        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=False,
        ):
            restored_metadata = _restore_metadata(metadata)
            if not metadata_matches(restored_metadata, query.metadata_filter):
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    document_id=str(metadata["document_id"]),
                    chunk_index=int(metadata["chunk_index"]),
                    text=str(text),
                    token_count=int(metadata["token_count"]),
                    metadata=restored_metadata,
                    embedding_model=str(metadata["embedding_model"]),
                    score=round(1.0 - float(distance), 6),
                )
            )

        latency_ms = round((perf_counter() - started_at) * 1000, 3)
        return RetrievalResult(
            query=query.query,
            top_k=query.top_k,
            strategy=query.strategy,
            backend=self.backend_name,
            embedding_model=self._embedding_provider.model_name,
            latency_ms=latency_ms,
            results=chunks[: query.top_k],
        )


def _chroma_metadata(chunk: ChunkRecord) -> dict[str, str | int | float | bool]:
    metadata = {
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "token_count": chunk.token_count,
        "embedding_model": chunk.embedding_model,
    }
    for key, value in chunk.metadata.items():
        if isinstance(value, str | int | float | bool):
            metadata[key] = value
        else:
            metadata[key] = json.dumps(value, sort_keys=True)
    return metadata


def _restore_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    restored = dict(metadata)
    restored.pop("document_id", None)
    restored.pop("chunk_index", None)
    restored.pop("token_count", None)
    restored.pop("embedding_model", None)
    for key, value in list(restored.items()):
        if isinstance(value, str) and value[:1] in {"[", "{"}:
            try:
                restored[key] = json.loads(value)
            except json.JSONDecodeError:
                restored[key] = value
    return restored


def _chroma_where(metadata_filter: dict[str, Any]) -> dict[str, Any] | None:
    if not metadata_filter:
        return None

    where: dict[str, Any] = {}
    for key, value in metadata_filter.items():
        if isinstance(value, str | int | float | bool):
            where[key] = value
        elif isinstance(value, list | tuple | set):
            where[key] = {"$in": list(value)}
        elif isinstance(value, dict) and "$eq" in value:
            where[key] = value["$eq"]
    return where or None
