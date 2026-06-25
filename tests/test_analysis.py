from __future__ import annotations

from fastapi.testclient import TestClient

from app.analysis.models import FailureThresholds
from app.analysis.repositories import InMemoryFailureAnalysisRepository
from app.analysis.service import ErrorAnalysisService
from app.core.dependencies import (
    get_failure_analysis_repository,
    get_leaderboard_repository,
    get_trace_repository,
)
from app.main import app
from app.retrieval.leaderboard import InMemoryLeaderboardRepository
from app.retrieval.models import (
    GroundTruthQuery,
    RetrievalEvaluationSummary,
    RetrievalQueryMetrics,
)
from app.storage.registry import get_document_repository
from app.traces.repositories import InMemoryTraceRepository


def test_error_analysis_service_classifies_metric_failures() -> None:
    repository = InMemoryFailureAnalysisRepository()
    service = ErrorAnalysisService(
        repository=repository,
        thresholds=FailureThresholds(
            precision_at_k=0.8,
            recall_at_k=0.8,
            mrr=0.8,
            ndcg=0.8,
            latency_ms=50.0,
            avg_similarity_score=0.6,
        ),
    )
    summary = RetrievalEvaluationSummary(
        run_id="experiment-1",
        dataset_size=1,
        top_k=3,
        strategy="cosine",
        backend="memory",
        embedding_model="local-hash-32d",
        precision_at_k=0.3333,
        recall_at_k=0.3333,
        mrr=0.0,
        ndcg=0.25,
        avg_latency_ms=75.0,
        per_query=[
            RetrievalQueryMetrics(
                query_id="q-policy",
                precision_at_k=0.3333,
                recall_at_k=0.3333,
                mrr=0.0,
                ndcg=0.25,
                latency_ms=75.0,
                retrieved_count=3,
                relevant_count=2,
                avg_similarity_score=0.2,
            )
        ],
    )

    failures = service.analyze_retrieval_evaluation(
        summary=summary,
        records=[
            GroundTruthQuery(
                query_id="q-policy",
                query="What is the refund policy?",
                relevant_document_ids={"doc-policy"},
            )
        ],
    )

    assert {failure.failure_type for failure in failures} == {
        "LOW_PRECISION",
        "LOW_RECALL",
        "LOW_MRR",
        "LOW_NDCG",
        "HIGH_LATENCY",
        "LOW_SIMILARITY",
    }
    assert all(failure.query == "What is the refund policy?" for failure in failures)
    assert all(failure.experiment_id == "experiment-1" for failure in failures)
    assert all(failure.recommendation for failure in failures)
    precision_failure = next(
        failure for failure in failures if failure.failure_type == "LOW_PRECISION"
    )
    assert "Reduce chunk size" in precision_failure.recommendation
    assert repository.list_failures(limit=10)


def test_error_analysis_service_does_not_flag_threshold_boundaries() -> None:
    repository = InMemoryFailureAnalysisRepository()
    service = ErrorAnalysisService(
        repository=repository,
        thresholds=FailureThresholds(
            precision_at_k=0.7,
            recall_at_k=0.7,
            mrr=0.7,
            ndcg=0.7,
            latency_ms=100.0,
            avg_similarity_score=0.5,
        ),
    )
    summary = RetrievalEvaluationSummary(
        run_id="experiment-pass",
        dataset_size=1,
        top_k=1,
        strategy="cosine",
        backend="memory",
        embedding_model="local-hash-32d",
        precision_at_k=0.7,
        recall_at_k=0.7,
        mrr=0.7,
        ndcg=0.7,
        avg_latency_ms=100.0,
        per_query=[
            RetrievalQueryMetrics(
                query_id="q-pass",
                precision_at_k=0.7,
                recall_at_k=0.7,
                mrr=0.7,
                ndcg=0.7,
                latency_ms=100.0,
                retrieved_count=1,
                relevant_count=1,
                avg_similarity_score=0.5,
            )
        ],
    )

    failures = service.analyze_retrieval_evaluation(summary=summary)

    assert failures == []
    assert repository.list_failures() == []


def test_analysis_failures_api_returns_failures_from_retrieval_evaluation() -> None:
    get_document_repository().clear()
    analysis_repository = InMemoryFailureAnalysisRepository()
    leaderboard_repository = InMemoryLeaderboardRepository()
    trace_repository = InMemoryTraceRepository()
    app.dependency_overrides[get_failure_analysis_repository] = lambda: analysis_repository
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
                "dataset_name": "analysis-api",
                "run_name": "analysis-api-run",
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
        assert eval_response.json()["per_query"][0]["precision_at_k"] == 0.5

        failures_response = client.get("/analysis/failures")
        assert failures_response.status_code == 200
        failures = failures_response.json()
        assert failures
        low_precision = next(
            failure for failure in failures if failure["failure_type"] == "LOW_PRECISION"
        )
        assert low_precision["query"] == "retrieval precision recall ranking"
        assert low_precision["metric_name"] == "precision_at_k"
        assert low_precision["metric_value"] == 0.5
        assert low_precision["threshold"] == 0.7
        assert "Reduce chunk size" in low_precision["recommendation"]
    finally:
        app.dependency_overrides.clear()
