from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from app.core.errors import DependencyUnavailableError
from app.rag.embeddings import EmbeddingProvider
from app.retrieval.models import RetrievalQuery, RetrievalResult, RetrievedChunk
from app.storage.models import ChunkRecord


class PgVectorRetriever:
    backend_name = "pgvector"

    def __init__(
        self,
        *,
        dsn: str,
        table_name: str,
        embedding_dimensions: int,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        try:
            import psycopg
        except ModuleNotFoundError as exc:
            raise DependencyUnavailableError(
                "psycopg is not installed. Install with `pip install -e '.[platform]'`."
            ) from exc

        self._psycopg = psycopg
        self._dsn = dsn
        self._table_name = _safe_identifier(table_name)
        self._embedding_dimensions = embedding_dimensions
        self._embedding_provider = embedding_provider
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with self._psycopg.connect(self._dsn) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    chunk_id text PRIMARY KEY,
                    document_id text NOT NULL,
                    chunk_index integer NOT NULL,
                    text text NOT NULL,
                    token_count integer NOT NULL,
                    metadata jsonb NOT NULL,
                    embedding_model text NOT NULL,
                    embedding vector({self._embedding_dimensions}) NOT NULL,
                    created_at timestamptz DEFAULT now()
                )
                """
            )
            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._table_name}_embedding_hnsw_idx
                ON {self._table_name}
                USING hnsw (embedding vector_cosine_ops)
                """
            )
            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._table_name}_metadata_gin_idx
                ON {self._table_name}
                USING gin (metadata)
                """
            )

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            return

        rows = [
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.chunk_index,
                chunk.text,
                chunk.token_count,
                json.dumps(chunk.metadata, sort_keys=True),
                chunk.embedding_model,
                _vector_literal(chunk.embedding),
            )
            for chunk in chunks
        ]
        with self._psycopg.connect(self._dsn) as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    f"""
                    INSERT INTO {self._table_name}
                        (chunk_id, document_id, chunk_index, text, token_count,
                         metadata, embedding_model, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s::vector)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        chunk_index = EXCLUDED.chunk_index,
                        text = EXCLUDED.text,
                        token_count = EXCLUDED.token_count,
                        metadata = EXCLUDED.metadata,
                        embedding_model = EXCLUDED.embedding_model,
                        embedding = EXCLUDED.embedding
                    """,
                    rows,
                )

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        if not query.query.strip():
            raise ValueError("query must not be blank")
        if query.top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        started_at = perf_counter()
        query_embedding = self._embedding_provider.embed_documents([query.query])[0]
        where_clause, parameters = _metadata_predicate(query.metadata_filter)
        parameters.extend([_vector_literal(query_embedding), query.top_k])

        with self._psycopg.connect(self._dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        chunk_id,
                        document_id,
                        chunk_index,
                        text,
                        token_count,
                        metadata,
                        embedding_model,
                        1 - (embedding <=> %s::vector) AS score
                    FROM {self._table_name}
                    {where_clause}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    [*parameters[:-2], parameters[-2], parameters[-2], parameters[-1]],
                )
                rows = cursor.fetchall()

        latency_ms = round((perf_counter() - started_at) * 1000, 3)
        return RetrievalResult(
            query=query.query,
            top_k=query.top_k,
            strategy=query.strategy,
            backend=self.backend_name,
            embedding_model=self._embedding_provider.model_name,
            latency_ms=latency_ms,
            results=[
                RetrievedChunk(
                    chunk_id=str(row[0]),
                    document_id=str(row[1]),
                    chunk_index=int(row[2]),
                    text=str(row[3]),
                    token_count=int(row[4]),
                    metadata=dict(row[5]),
                    embedding_model=str(row[6]),
                    score=round(float(row[7]), 6),
                )
                for row in rows
            ],
        )


def _metadata_predicate(metadata_filter: dict[str, Any]) -> tuple[str, list[Any]]:
    if not metadata_filter:
        return "", []
    exact_matches = {
        key: value
        for key, value in metadata_filter.items()
        if isinstance(value, str | int | float | bool)
    }
    if not exact_matches:
        return "", []
    return "WHERE metadata @> %s::jsonb", [json.dumps(exact_matches, sort_keys=True)]


def _vector_literal(vector: list[float]) -> str:
    return f"[{','.join(str(float(value)) for value in vector)}]"


def _safe_identifier(value: str) -> str:
    if not value.replace("_", "").isalnum() or value[0].isdigit():
        raise ValueError("table_name must be a safe SQL identifier")
    return value
