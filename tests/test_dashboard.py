from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from app.core.dependencies import (
    get_leaderboard_repository,
    get_quality_gate_repository,
    get_trace_repository,
)
from app.dashboard.models import DashboardFilters
from app.dashboard.repositories import CompositeDashboardRepository
from app.dashboard.service import DashboardService
from app.main import app
from app.quality.models import QualityGate
from app.quality.repositories import InMemoryQualityGateRepository
from app.retrieval.leaderboard import InMemoryLeaderboardRepository
from app.retrieval.models import LeaderboardEntry
from app.traces.models import Trace, TraceStatus
from app.traces.repositories import InMemoryTraceRepository


def _trace(*, trace_id: str, status: TraceStatus, latency_ms: float) -> Trace:
    return Trace(
        trace_id=trace_id,
        timestamp=datetime(2026, 6, 22, 12, 0, tzinfo=UTC),
        query=f"{trace_id} query",
        retriever_name="memory",
        embedding_model="local-hash-32d",
        retrieved_chunks=[],
        similarity_scores=[],
        retrieval_latency_ms=latency_ms,
        generation_latency_ms=0.0,
        total_latency_ms=latency_ms,
        status=status,
    )


def _leaderboard_entry(
    *,
    run_id: str,
    embedding_model: str = "local-hash-32d",
    backend: str = "memory",
    chunk_size_tokens: int | None = 48,
    precision_at_k: float,
    recall_at_k: float,
    mrr: float,
    ndcg: float,
) -> LeaderboardEntry:
    return LeaderboardEntry(
        run_id=run_id,
        name=f"{run_id}-name",
        dataset_name="unit-dataset",
        backend=backend,
        embedding_model=embedding_model,
        retrieval_strategy="cosine",
        top_k=5,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=8,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        mrr=mrr,
        ndcg=ndcg,
        avg_latency_ms=25.0,
    )


def _seed_dashboard_repositories() -> tuple[
    InMemoryTraceRepository,
    InMemoryLeaderboardRepository,
    InMemoryQualityGateRepository,
]:
    trace_repository = InMemoryTraceRepository()
    for trace in [
        _trace(trace_id="trace-1", status="completed", latency_ms=10.0),
        _trace(trace_id="trace-2", status="completed", latency_ms=20.0),
        _trace(trace_id="trace-3", status="failed", latency_ms=100.0),
    ]:
        trace_repository.save(trace)

    leaderboard_repository = InMemoryLeaderboardRepository()
    leaderboard_repository.record(
        _leaderboard_entry(
            run_id="run-a",
            precision_at_k=0.9,
            recall_at_k=0.7,
            mrr=0.8,
            ndcg=0.75,
        )
    )
    leaderboard_repository.record(
        _leaderboard_entry(
            run_id="run-b",
            embedding_model="local-hash-64d",
            backend="pgvector",
            chunk_size_tokens=96,
            precision_at_k=0.6,
            recall_at_k=0.95,
            mrr=0.9,
            ndcg=0.92,
        )
    )

    quality_repository = InMemoryQualityGateRepository()
    now = datetime(2026, 6, 22, 12, 30, tzinfo=UTC)
    quality_repository.save(
        QualityGate(
            experiment_id="run-a",
            status="FAILED",
            failed_checks=[],
            metrics_snapshot={},
            timestamp=now,
        )
    )
    quality_repository.save(
        QualityGate(
            experiment_id="run-b",
            status="PASSED",
            failed_checks=[],
            metrics_snapshot={},
            timestamp=now + timedelta(minutes=1),
        )
    )
    return trace_repository, leaderboard_repository, quality_repository


def test_dashboard_service_summarizes_traces_and_quality_gates() -> None:
    trace_repository, leaderboard_repository, quality_repository = _seed_dashboard_repositories()
    service = DashboardService(
        repository=CompositeDashboardRepository(
            trace_repository=trace_repository,
            leaderboard_repository=leaderboard_repository,
            quality_gate_repository=quality_repository,
        )
    )

    summary = service.summary()

    assert summary.total_traces == 3
    assert summary.successful_requests == 2
    assert summary.failed_requests == 1
    assert summary.success_rate == 0.6667
    assert summary.error_rate == 0.3333
    assert summary.avg_retrieval_latency_ms == 43.333
    assert summary.p95_retrieval_latency_ms == 100.0
    assert summary.failed_quality_gates == 1
    assert summary.total_quality_checks == 2


def test_dashboard_service_sorts_and_filters_leaderboard() -> None:
    trace_repository, leaderboard_repository, quality_repository = _seed_dashboard_repositories()
    service = DashboardService(
        repository=CompositeDashboardRepository(
            trace_repository=trace_repository,
            leaderboard_repository=leaderboard_repository,
            quality_gate_repository=quality_repository,
        )
    )

    entries = service.leaderboard(sort_by="precision_at_k", sort_order="desc", limit=2)
    filtered = service.leaderboard(
        sort_by="ndcg",
        sort_order="desc",
        limit=5,
        filters=DashboardFilters(
            embedding_model="local-hash-64d",
            retriever="pgvector",
            quality_gate_status="PASSED",
        ),
    )

    assert [entry.experiment_id for entry in entries] == ["run-a", "run-b"]
    assert entries[0].quality_gate_status == "FAILED"
    assert len(filtered) == 1
    assert filtered[0].experiment_id == "run-b"
    assert filtered[0].quality_gate_status == "PASSED"


def test_dashboard_apis_return_summary_leaderboard_and_experiments() -> None:
    trace_repository, leaderboard_repository, quality_repository = _seed_dashboard_repositories()
    app.dependency_overrides[get_trace_repository] = lambda: trace_repository
    app.dependency_overrides[get_leaderboard_repository] = lambda: leaderboard_repository
    app.dependency_overrides[get_quality_gate_repository] = lambda: quality_repository
    client = TestClient(app)

    try:
        summary_response = client.get("/dashboard/summary")
        assert summary_response.status_code == 200
        summary = summary_response.json()
        assert summary["total_traces"] == 3
        assert summary["failed_quality_gates"] == 1
        assert summary["total_quality_checks"] == 2

        leaderboard_response = client.get(
            "/dashboard/leaderboard",
            params={
                "limit": 1,
                "sort_by": "recall_at_k",
                "sort_order": "desc",
                "quality_gate_status": "PASSED",
            },
        )
        assert leaderboard_response.status_code == 200
        leaderboard = leaderboard_response.json()
        assert len(leaderboard) == 1
        assert leaderboard[0]["experiment_id"] == "run-b"
        assert leaderboard[0]["recall_at_k"] == 0.95
        assert leaderboard[0]["quality_gate_status"] == "PASSED"

        experiments_response = client.get(
            "/dashboard/experiments",
            params={"embedding_model": "local-hash-32d"},
        )
        assert experiments_response.status_code == 200
        experiments = experiments_response.json()
        assert len(experiments) == 1
        assert experiments[0] == {
            "experiment_id": "run-a",
            "embedding_model": "local-hash-32d",
            "chunk_size": 48,
            "retriever": "memory",
            "top_k": 5,
            "precision": 0.9,
            "recall": 0.7,
            "mrr": 0.8,
            "ndcg": 0.75,
            "quality_gate_status": "FAILED",
        }
    finally:
        app.dependency_overrides.clear()
