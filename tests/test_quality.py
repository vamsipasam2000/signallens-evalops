from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.dependencies import (
    get_leaderboard_repository,
    get_quality_gate_repository,
    get_trace_repository,
)
from app.main import app
from app.quality.models import QualityMetricsSnapshot, QualityThresholds
from app.quality.repositories import InMemoryQualityGateRepository
from app.quality.service import QualityGateService
from app.retrieval.leaderboard import InMemoryLeaderboardRepository
from app.storage.registry import get_document_repository
from app.traces.repositories import InMemoryTraceRepository


def test_quality_gate_service_fails_experiment_with_failed_checks() -> None:
    repository = InMemoryQualityGateRepository()
    service = QualityGateService(
        repository=repository,
        thresholds=QualityThresholds(
            precision_at_k=0.8,
            recall_at_k=0.75,
            mrr=0.7,
            ndcg=0.8,
            retrieval_latency_ms=50.0,
            similarity_score=0.6,
        ),
    )

    gate = service.evaluate_experiment(
        experiment_id="experiment-low-quality",
        metrics_snapshot=QualityMetricsSnapshot(
            precision_at_k=0.62,
            recall_at_k=0.8,
            mrr=0.7,
            ndcg=0.9,
            retrieval_latency_ms=80.0,
            similarity_score=0.4,
        ),
    )

    assert gate.status == "FAILED"
    assert [check.metric for check in gate.failed_checks] == [
        "precision_at_k",
        "retrieval_latency_ms",
        "similarity_score",
    ]
    precision_check = gate.failed_checks[0]
    assert precision_check.actual == 0.62
    assert precision_check.required == 0.8
    assert precision_check.reason == "LOW_PRECISION"
    assert "Review chunking strategy" in precision_check.recommendation
    assert repository.list_checks()[0].gate_id == gate.gate_id


def test_quality_gate_service_passes_threshold_boundaries() -> None:
    repository = InMemoryQualityGateRepository()
    service = QualityGateService(
        repository=repository,
        thresholds=QualityThresholds(
            precision_at_k=0.8,
            recall_at_k=0.8,
            mrr=0.8,
            ndcg=0.8,
            retrieval_latency_ms=100.0,
            similarity_score=0.5,
        ),
    )

    gate = service.evaluate_experiment(
        experiment_id="experiment-pass",
        metrics_snapshot=QualityMetricsSnapshot(
            precision_at_k=0.8,
            recall_at_k=0.8,
            mrr=0.8,
            ndcg=0.8,
            retrieval_latency_ms=100.0,
            similarity_score=0.5,
        ),
    )

    assert gate.status == "PASSED"
    assert gate.failed_checks == []


def test_quality_check_api_records_and_lists_gate() -> None:
    repository = InMemoryQualityGateRepository()
    app.dependency_overrides[get_quality_gate_repository] = lambda: repository
    client = TestClient(app)

    try:
        response = client.post(
            "/quality/check",
            json={
                "experiment_id": "manual-experiment",
                "precision_at_k": 0.62,
                "recall_at_k": 0.9,
                "mrr": 0.85,
                "ndcg": 0.9,
                "retrieval_latency_ms": 25.0,
                "similarity_score": 0.7,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "FAILED"
        assert body["failed_checks"] == [
            {
                "metric": "precision_at_k",
                "actual": 0.62,
                "required": 0.8,
                "reason": "LOW_PRECISION",
                "recommendation": (
                    "Review chunking strategy; evaluate embedding model; "
                    "apply metadata filtering."
                ),
            }
        ]
        assert body["metrics_snapshot"]["precision_at_k"] == 0.62

        list_response = client.get("/quality/checks")
        assert list_response.status_code == 200
        checks = list_response.json()
        assert len(checks) == 1
        assert checks[0]["gate_id"] == body["gate_id"]
        assert checks[0]["experiment_id"] == "manual-experiment"
    finally:
        app.dependency_overrides.clear()


def test_retrieval_evaluation_records_quality_gate() -> None:
    get_document_repository().clear()
    quality_repository = InMemoryQualityGateRepository()
    leaderboard_repository = InMemoryLeaderboardRepository()
    trace_repository = InMemoryTraceRepository()
    app.dependency_overrides[get_quality_gate_repository] = lambda: quality_repository
    app.dependency_overrides[get_leaderboard_repository] = lambda: leaderboard_repository
    app.dependency_overrides[get_trace_repository] = lambda: trace_repository
    client = TestClient(app)

    try:
        ingest_response = client.post(
            "/v1/rag/ingest",
            json={
                "document_name": "ranking.md",
                "content": "Precision and recall measure retrieval ranking quality.",
                "content_type": "text/plain",
                "metadata": {"domain": "evaluation"},
                "chunk_size_tokens": 20,
                "chunk_overlap_tokens": 2,
            },
        )
        assert ingest_response.status_code == 200
        document_id = ingest_response.json()["document_id"]

        eval_response = client.post(
            "/evaluate/retrieval",
            json={
                "dataset_name": "quality-api",
                "run_name": "quality-api-run",
                "top_k": 2,
                "ground_truth": [
                    {
                        "query_id": "ranking",
                        "query": "retrieval precision recall ranking",
                        "relevant_document_ids": [document_id],
                        "metadata_filter": {"domain": "evaluation"},
                    }
                ],
            },
        )
        assert eval_response.status_code == 200
        run_id = eval_response.json()["run_id"]

        checks_response = client.get("/quality/checks")
        assert checks_response.status_code == 200
        checks = checks_response.json()
        assert len(checks) == 1
        assert checks[0]["experiment_id"] == run_id
        assert checks[0]["status"] == "FAILED"
        assert checks[0]["failed_checks"][0]["metric"] == "precision_at_k"
    finally:
        app.dependency_overrides.clear()
