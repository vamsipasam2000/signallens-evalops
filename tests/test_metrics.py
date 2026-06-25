from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.core.dependencies import get_trace_repository
from app.main import app
from app.metrics.repositories import TraceBackedMetricsRepository
from app.metrics.service import MetricsService
from app.traces.models import Trace, TraceChunk, TraceStatus
from app.traces.repositories import InMemoryTraceRepository


def _trace(
    *,
    trace_id: str,
    status: TraceStatus,
    retrieval_latency_ms: float,
    retrieved_chunk_scores: list[float],
) -> Trace:
    return Trace(
        trace_id=trace_id,
        timestamp=datetime(2026, 6, 19, 12, 0, tzinfo=UTC),
        query=f"{trace_id} query",
        request_payload={"trace_id": trace_id},
        retriever_name="memory",
        embedding_model="local-hash-32d",
        retrieved_chunks=[
            TraceChunk(
                chunk_id=f"{trace_id}-chunk-{index}",
                document_id=f"{trace_id}-doc",
                chunk_index=index,
                text=f"{trace_id} chunk {index}",
                metadata={},
                score=score,
            )
            for index, score in enumerate(retrieved_chunk_scores)
        ],
        similarity_scores=retrieved_chunk_scores,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=0.0,
        total_latency_ms=retrieval_latency_ms,
        status=status,
        error_message="boom" if status == "failed" else None,
    )


def test_metrics_service_summarizes_trace_health() -> None:
    trace_repository = InMemoryTraceRepository()
    for trace in [
        _trace(
            trace_id="trace-1",
            status="completed",
            retrieval_latency_ms=10.0,
            retrieved_chunk_scores=[0.8, 0.6],
        ),
        _trace(
            trace_id="trace-2",
            status="completed",
            retrieval_latency_ms=30.0,
            retrieved_chunk_scores=[0.4],
        ),
        _trace(
            trace_id="trace-3",
            status="failed",
            retrieval_latency_ms=100.0,
            retrieved_chunk_scores=[],
        ),
    ]:
        trace_repository.save(trace)
    service = MetricsService(
        repository=TraceBackedMetricsRepository(trace_repository=trace_repository)
    )

    snapshot = service.summarize()

    assert snapshot.total_requests == 3
    assert snapshot.successful_requests == 2
    assert snapshot.failed_requests == 1
    assert snapshot.success_rate == 0.6667
    assert snapshot.error_rate == 0.3333
    assert snapshot.avg_retrieval_latency_ms == 46.667
    assert snapshot.p95_retrieval_latency_ms == 100.0
    assert snapshot.avg_chunks_returned == 1.0
    assert snapshot.avg_similarity_score == 0.6
    assert snapshot.request_volume == 3
    assert snapshot.trace_volume == 3
    assert snapshot.failure_count == 1


def test_metrics_service_returns_zero_snapshot_without_traces() -> None:
    service = MetricsService(
        repository=TraceBackedMetricsRepository(
            trace_repository=InMemoryTraceRepository(),
        )
    )

    snapshot = service.summarize()

    assert snapshot.total_requests == 0
    assert snapshot.successful_requests == 0
    assert snapshot.failed_requests == 0
    assert snapshot.success_rate == 0.0
    assert snapshot.error_rate == 0.0
    assert snapshot.avg_retrieval_latency_ms == 0.0
    assert snapshot.p95_retrieval_latency_ms == 0.0
    assert snapshot.avg_chunks_returned == 0.0
    assert snapshot.avg_similarity_score == 0.0
    assert snapshot.request_volume == 0
    assert snapshot.trace_volume == 0
    assert snapshot.failure_count == 0


def test_metrics_summary_api_derives_metrics_from_stored_traces() -> None:
    trace_repository = InMemoryTraceRepository()
    trace_repository.save(
        _trace(
            trace_id="trace-api-1",
            status="completed",
            retrieval_latency_ms=20.0,
            retrieved_chunk_scores=[0.9],
        )
    )
    trace_repository.save(
        _trace(
            trace_id="trace-api-2",
            status="failed",
            retrieval_latency_ms=80.0,
            retrieved_chunk_scores=[],
        )
    )
    app.dependency_overrides[get_trace_repository] = lambda: trace_repository
    client = TestClient(app)

    try:
        response = client.get("/metrics/summary")
        assert response.status_code == 200
        body = response.json()
        assert body["request_volume"] == 2
        assert body["trace_volume"] == 2
        assert body["failure_count"] == 1
        assert body["success_rate"] == 0.5
        assert body["error_rate"] == 0.5
        assert body["avg_retrieval_latency_ms"] == 50.0
        assert body["p95_retrieval_latency_ms"] == 80.0
        assert body["avg_similarity_score"] == 0.9
    finally:
        app.dependency_overrides.clear()
