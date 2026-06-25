from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import get_trace_repository
from app.core.errors import RetrievalValidationError
from app.main import app
from app.rag.embeddings import HashEmbeddingProvider
from app.retrieval.models import RetrievalQuery
from app.retrieval.retrievers import RepositoryRetriever
from app.retrieval.service import RetrievalService
from app.storage.models import ChunkRecord, DocumentRecord
from app.storage.registry import get_document_repository
from app.storage.repositories import InMemoryDocumentRepository
from app.traces.models import Trace, TraceChunk, TraceStatus
from app.traces.repositories import InMemoryTraceRepository
from app.traces.service import TraceListFilters, TraceService


def _seed_repository() -> InMemoryDocumentRepository:
    repository = InMemoryDocumentRepository()
    provider = HashEmbeddingProvider(dimensions=32)
    text = "Trace collection records retrieved chunks and similarity scores."
    repository.save_document(
        DocumentRecord(
            document_id="doc-traces",
            name="traces.md",
            content_type="text/markdown",
            source="unit-test",
            metadata={"domain": "observability"},
            chunk_count=1,
            token_count=len(text.split()),
            ingestion_latency_ms=0.0,
            parser_version="test",
            chunker_version="test",
            embedding_model=provider.model_name,
        )
    )
    repository.save_chunks(
        [
            ChunkRecord(
                chunk_id="doc-traces-chunk-0",
                document_id="doc-traces",
                chunk_index=0,
                text=text,
                token_count=len(text.split()),
                metadata={"domain": "observability"},
                embedding=provider.embed_documents([text])[0],
                embedding_model=provider.model_name,
            )
        ]
    )
    return repository


def _trace(
    *,
    trace_id: str,
    timestamp: datetime,
    query: str,
    retriever_name: str,
    embedding_model: str,
    status: TraceStatus,
    retrieval_latency_ms: float,
    total_latency_ms: float,
) -> Trace:
    return Trace(
        trace_id=trace_id,
        timestamp=timestamp,
        query=query,
        retriever_name=retriever_name,
        embedding_model=embedding_model,
        retrieved_chunks=[],
        similarity_scores=[],
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=0.0,
        total_latency_ms=total_latency_ms,
        status=status,
    )


def test_in_memory_trace_repository_saves_and_gets_copy() -> None:
    repository = InMemoryTraceRepository()
    trace = Trace(
        query="trace storage",
        retriever_name="memory",
        embedding_model="local-hash",
        retrieved_chunks=[
            TraceChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                chunk_index=0,
                text="trace text",
                metadata={"domain": "observability"},
                score=0.91,
            )
        ],
        similarity_scores=[0.91],
        retrieval_latency_ms=3.2,
        generation_latency_ms=0.0,
        total_latency_ms=3.2,
        status="completed",
    )

    stored = repository.save(trace)
    fetched = repository.get(stored.trace_id)

    assert fetched is not None
    assert fetched.trace_id == stored.trace_id
    assert fetched.retrieved_chunks[0].chunk_id == "chunk-1"
    fetched.retrieved_chunks[0].metadata["domain"] = "mutated"
    assert repository.get(stored.trace_id).retrieved_chunks[0].metadata["domain"] == "observability"


def test_trace_service_lists_filters_sorts_and_paginates() -> None:
    repository = InMemoryTraceRepository()
    base_time = datetime(2026, 6, 19, 12, 0, tzinfo=UTC)
    first = repository.save(
        _trace(
            trace_id="trace-1",
            timestamp=base_time,
            query="first retrieval",
            retriever_name="memory",
            embedding_model="local-hash-32d",
            status="completed",
            retrieval_latency_ms=5.0,
            total_latency_ms=5.0,
        )
    )
    second = repository.save(
        _trace(
            trace_id="trace-2",
            timestamp=base_time + timedelta(minutes=5),
            query="second retrieval",
            retriever_name="pgvector",
            embedding_model="local-hash-64d",
            status="failed",
            retrieval_latency_ms=2.0,
            total_latency_ms=2.0,
        )
    )
    repository.save(
        _trace(
            trace_id="trace-3",
            timestamp=base_time + timedelta(minutes=10),
            query="third retrieval",
            retriever_name="memory",
            embedding_model="local-hash-32d",
            status="completed",
            retrieval_latency_ms=9.0,
            total_latency_ms=9.0,
        )
    )
    service = TraceService(repository=repository)

    result = service.list(
        filters=TraceListFilters(
            status="completed",
            retriever_name="memory",
            embedding_model="local-hash-32d",
            start_date=base_time - timedelta(minutes=1),
            end_date=base_time + timedelta(minutes=11),
        ),
        limit=1,
        offset=1,
        sort_by="retrieval_latency_ms",
        sort_order="asc",
    )

    assert result.total == 2
    assert result.limit == 1
    assert result.offset == 1
    assert result.items[0].trace_id == "trace-3"
    assert result.items[0].query == "third retrieval"
    assert result.items[0].retrieval_latency_ms == 9.0

    failed_result = service.list(
        filters=TraceListFilters(status="failed"),
        limit=10,
        offset=0,
        sort_by="timestamp",
        sort_order="desc",
    )
    assert failed_result.total == 1
    assert failed_result.items[0].trace_id == second.trace_id

    stored_first = repository.get(first.trace_id)
    assert stored_first is not None
    assert stored_first.query == "first retrieval"


def test_retrieval_service_records_completed_trace() -> None:
    trace_repository = InMemoryTraceRepository()
    provider = HashEmbeddingProvider(dimensions=32)
    service = RetrievalService(
        retriever=RepositoryRetriever(
            repository=_seed_repository(),
            embedding_provider=provider,
        ),
        trace_repository=trace_repository,
        embedding_model=provider.model_name,
    )

    result = service.retrieve(
        RetrievalQuery(
            query="retrieved chunks similarity scores",
            top_k=1,
            metadata_filter={"domain": "observability"},
        )
    )

    assert result.trace_id is not None
    trace = trace_repository.get(result.trace_id)
    assert trace is not None
    assert trace.status == "completed"
    assert trace.request_payload == {
        "query": "retrieved chunks similarity scores",
        "top_k": 1,
        "metadata_filter": {"domain": "observability"},
        "strategy": "cosine",
    }
    assert trace.query == result.query
    assert trace.retriever_name == "memory"
    assert trace.embedding_model == provider.model_name
    assert trace.retrieved_chunks[0].chunk_id == result.results[0].chunk_id
    assert trace.similarity_scores == [result.results[0].score]
    assert trace.retrieval_latency_ms == result.latency_ms
    assert trace.generation_latency_ms == 0.0
    assert trace.total_latency_ms == result.latency_ms
    assert trace.error_message is None


def test_retrieval_service_records_failed_trace() -> None:
    trace_repository = InMemoryTraceRepository()
    provider = HashEmbeddingProvider(dimensions=32)
    service = RetrievalService(
        retriever=RepositoryRetriever(
            repository=_seed_repository(),
            embedding_provider=provider,
        ),
        trace_repository=trace_repository,
        embedding_model=provider.model_name,
    )

    with pytest.raises(RetrievalValidationError, match="top_k"):
        service.retrieve(RetrievalQuery(query="trace failure", top_k=0))

    traces = trace_repository.list_all()
    assert len(traces) == 1
    assert traces[0].status == "failed"
    assert traces[0].request_payload == {
        "query": "trace failure",
        "top_k": 0,
        "metadata_filter": {},
        "strategy": "cosine",
    }
    assert traces[0].query == "trace failure"
    assert traces[0].retrieved_chunks == []
    assert traces[0].similarity_scores == []
    assert traces[0].error_message == "top_k must be greater than zero"


def test_trace_api_create_get_and_retrieval_trace() -> None:
    trace_repository = InMemoryTraceRepository()
    get_document_repository().clear()
    app.dependency_overrides[get_trace_repository] = lambda: trace_repository
    client = TestClient(app)

    try:
        create_response = client.post(
            "/traces",
            json={
                "query": "manual trace",
                "retriever_name": "memory",
                "embedding_model": "local-hash",
                "retrieved_chunks": [],
                "similarity_scores": [],
                "retrieval_latency_ms": 1.2,
                "generation_latency_ms": 0.0,
                "total_latency_ms": 1.2,
                "status": "completed",
            },
        )
        assert create_response.status_code == 200
        manual_trace_id = create_response.json()["trace_id"]

        get_response = client.get(f"/traces/{manual_trace_id}")
        assert get_response.status_code == 200
        assert get_response.json()["query"] == "manual trace"

        missing_response = client.get("/traces/missing")
        assert missing_response.status_code == 404

        ingest_response = client.post(
            "/v1/rag/ingest",
            json={
                "document_name": "trace-retrieval.md",
                "content": "Trace collection records retrieved chunks and similarity scores.",
                "content_type": "text/plain",
                "metadata": {"domain": "observability"},
                "chunk_size_tokens": 20,
                "chunk_overlap_tokens": 2,
            },
        )
        assert ingest_response.status_code == 200

        retrieve_response = client.post(
            "/retrieve",
            json={
                "query": "retrieved chunks similarity scores",
                "top_k": 1,
                "metadata_filter": {"domain": "observability"},
            },
        )
        assert retrieve_response.status_code == 200
        generated_trace_id = retrieve_response.json()["trace_id"]

        generated_trace_response = client.get(f"/traces/{generated_trace_id}")
        assert generated_trace_response.status_code == 200
        generated_trace = generated_trace_response.json()
        assert generated_trace["status"] == "completed"
        assert generated_trace["request_payload"] == {
            "endpoint": "/retrieve",
            "body": {
                "query": "retrieved chunks similarity scores",
                "top_k": 1,
                "metadata_filter": {"domain": "observability"},
                "strategy": "cosine",
            },
        }
        assert generated_trace["retrieved_chunks"][0]["chunk_id"]
        assert generated_trace["similarity_scores"]
    finally:
        app.dependency_overrides.clear()


def test_trace_api_lists_with_filters_sorting_and_pagination() -> None:
    trace_repository = InMemoryTraceRepository()
    base_time = datetime(2026, 6, 19, 9, 0, tzinfo=UTC)
    trace_repository.save(
        _trace(
            trace_id="trace-fast",
            timestamp=base_time,
            query="fast retrieval",
            retriever_name="memory",
            embedding_model="local-hash-32d",
            status="completed",
            retrieval_latency_ms=4.0,
            total_latency_ms=4.0,
        )
    )
    trace_repository.save(
        _trace(
            trace_id="trace-slow",
            timestamp=base_time + timedelta(hours=1),
            query="slow retrieval",
            retriever_name="memory",
            embedding_model="local-hash-32d",
            status="completed",
            retrieval_latency_ms=12.0,
            total_latency_ms=12.0,
        )
    )
    trace_repository.save(
        _trace(
            trace_id="trace-failed",
            timestamp=base_time + timedelta(hours=2),
            query="failed retrieval",
            retriever_name="pgvector",
            embedding_model="local-hash-64d",
            status="failed",
            retrieval_latency_ms=1.0,
            total_latency_ms=1.0,
        )
    )
    app.dependency_overrides[get_trace_repository] = lambda: trace_repository
    client = TestClient(app)

    try:
        response = client.get(
            "/traces",
            params={
                "status": "completed",
                "retriever_name": "memory",
                "embedding_model": "local-hash-32d",
                "start_date": (base_time - timedelta(minutes=1)).isoformat(),
                "end_date": (base_time + timedelta(hours=2)).isoformat(),
                "sort_by": "retrieval_latency_ms",
                "sort_order": "desc",
                "limit": 1,
                "offset": 0,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert body["limit"] == 1
        assert body["offset"] == 0
        assert body["sort_by"] == "retrieval_latency_ms"
        assert body["sort_order"] == "desc"
        assert len(body["items"]) == 1
        item = body["items"][0]
        assert item["trace_id"] == "trace-slow"
        assert item["timestamp"].startswith("2026-06-19T10:00:00")
        assert item["query"] == "slow retrieval"
        assert item["retriever_name"] == "memory"
        assert item["embedding_model"] == "local-hash-32d"
        assert item["status"] == "completed"
        assert item["retrieval_latency_ms"] == 12.0
        assert item["total_latency_ms"] == 12.0
        assert "retrieved_chunks" not in body["items"][0]
        assert "similarity_scores" not in body["items"][0]

        detail_response = client.get("/traces/trace-slow")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["query"] == "slow retrieval"
        assert detail["request_payload"] == {}
        assert detail["retrieved_chunks"] == []
        assert detail["similarity_scores"] == []
        assert detail["retrieval_latency_ms"] == 12.0
        assert detail["total_latency_ms"] == 12.0
        assert detail["status"] == "completed"
        assert detail["error_message"] is None
    finally:
        app.dependency_overrides.clear()
