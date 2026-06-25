from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.dependencies import get_leaderboard_repository, get_trace_repository
from app.main import app
from app.rag.embeddings import HashEmbeddingProvider
from app.ranking.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    ndcg_at_k_from_flags,
    precision_at_k,
    recall_at_k,
    recall_at_k_from_flags,
)
from app.retrieval.benchmark import RetrievalBenchmarkSuite
from app.retrieval.evaluation import RetrievalEvaluationService
from app.retrieval.leaderboard import InMemoryLeaderboardRepository
from app.retrieval.models import BenchmarkConfig, GroundTruthQuery, RetrievalQuery
from app.retrieval.retrievers import RepositoryRetriever
from app.retrieval.service import RetrievalService
from app.storage.models import ChunkRecord, DocumentRecord
from app.storage.registry import get_document_repository
from app.storage.repositories import InMemoryDocumentRepository
from app.traces.repositories import InMemoryTraceRepository


def _seed_repository() -> tuple[InMemoryDocumentRepository, str, str]:
    repository = InMemoryDocumentRepository()
    provider = HashEmbeddingProvider(dimensions=32)
    documents = [
        (
            "doc-ranking",
            "ranking.md",
            "Precision at K and Recall at K measure retrieval ranking quality.",
            {"domain": "evaluation", "topic": "ranking"},
        ),
        (
            "doc-storage",
            "pgvector.md",
            "pgvector stores embeddings in PostgreSQL with vector cosine indexes.",
            {"domain": "storage", "backend": "pgvector"},
        ),
    ]
    for document_id, name, text, metadata in documents:
        repository.save_document(
            DocumentRecord(
                document_id=document_id,
                name=name,
                content_type="text/markdown",
                source="unit-test",
                metadata=metadata,
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
                    chunk_id=f"{document_id}-chunk-0",
                    document_id=document_id,
                    chunk_index=0,
                    text=text,
                    token_count=len(text.split()),
                    metadata=metadata,
                    embedding=provider.embed_documents([text])[0],
                    embedding_model=provider.model_name,
                )
            ]
        )
    return repository, "doc-ranking", "doc-storage"


def test_ranking_metrics_at_k() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"b", "c"}

    assert precision_at_k(retrieved, relevant, 3) == 0.6667
    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert mean_reciprocal_rank(retrieved, relevant) == 0.5
    assert ndcg_at_k(retrieved, relevant, 3) == 0.6934
    assert recall_at_k_from_flags([True, True], total_relevant=1, k=2) == 1.0
    assert ndcg_at_k_from_flags([True, True], total_relevant=1, k=2) == 1.0


def test_repository_retriever_filters_metadata_and_tracks_latency() -> None:
    repository, _, storage_doc_id = _seed_repository()
    provider = HashEmbeddingProvider(dimensions=32)
    service = RetrievalService(
        retriever=RepositoryRetriever(
            repository=repository,
            embedding_provider=provider,
        )
    )

    result = service.retrieve(
        RetrievalQuery(
            query="PostgreSQL vector cosine index",
            top_k=1,
            metadata_filter={"backend": "pgvector"},
        )
    )

    assert result.backend == "memory"
    assert result.latency_ms >= 0
    assert len(result.results) == 1
    assert result.results[0].document_id == storage_doc_id
    assert result.results[0].score > 0


def test_retrieval_evaluation_service_scores_ground_truth() -> None:
    repository, ranking_doc_id, _ = _seed_repository()
    provider = HashEmbeddingProvider(dimensions=32)
    retrieval_service = RetrievalService(
        retriever=RepositoryRetriever(
            repository=repository,
            embedding_provider=provider,
        )
    )
    evaluator = RetrievalEvaluationService(
        retrieval_service=retrieval_service,
        backend="memory",
        embedding_model=provider.model_name,
    )

    summary = evaluator.evaluate(
        records=[
            GroundTruthQuery(
                query_id="ranking",
                query="precision recall retrieval ranking",
                relevant_document_ids={ranking_doc_id},
                metadata_filter={"domain": "evaluation"},
            )
        ],
        top_k=1,
        strategy="cosine",
    )

    assert summary.dataset_size == 1
    assert summary.precision_at_k == 1.0
    assert summary.recall_at_k == 1.0
    assert summary.mrr == 1.0
    assert summary.ndcg == 1.0


def test_retrieval_api_retrieve_evaluate_and_leaderboard() -> None:
    get_document_repository().clear()
    leaderboard_repository = InMemoryLeaderboardRepository()
    trace_repository = InMemoryTraceRepository()
    app.dependency_overrides[get_leaderboard_repository] = lambda: leaderboard_repository
    app.dependency_overrides[get_trace_repository] = lambda: trace_repository
    client = TestClient(app)

    try:
        ingest_response = client.post(
            "/v1/rag/ingest",
            json={
                "document_name": "ranking.md",
                "content": "Precision at K and Recall at K measure retrieval ranking quality.",
                "content_type": "text/plain",
                "metadata": {"domain": "evaluation"},
                "chunk_size_tokens": 20,
                "chunk_overlap_tokens": 2,
            },
        )
        assert ingest_response.status_code == 200
        ranking_document_id = ingest_response.json()["document_id"]

        client.post(
            "/v1/rag/ingest",
            json={
                "document_name": "storage.md",
                "content": "ChromaDB stores vector embeddings in persistent collections.",
                "content_type": "text/plain",
                "metadata": {"domain": "storage"},
                "chunk_size_tokens": 20,
                "chunk_overlap_tokens": 2,
            },
        )

        retrieve_response = client.post(
            "/retrieve",
            json={
                "query": "retrieval precision recall ranking",
                "top_k": 1,
                "metadata_filter": {"domain": "evaluation"},
            },
        )
        assert retrieve_response.status_code == 200
        retrieve_body = retrieve_response.json()
        assert retrieve_body["results"][0]["document_id"] == ranking_document_id

        retrieve_trace = trace_repository.get(retrieve_body["trace_id"])
        assert retrieve_trace is not None
        assert retrieve_trace.status == "completed"
        assert retrieve_trace.request_payload["endpoint"] == "/retrieve"
        assert retrieve_trace.request_payload["body"]["query"] == (
            "retrieval precision recall ranking"
        )
        assert retrieve_trace.retrieved_chunks[0].document_id == ranking_document_id
        assert retrieve_trace.similarity_scores == [retrieve_body["results"][0]["score"]]
        assert retrieve_trace.retrieval_latency_ms == retrieve_body["latency_ms"]

        eval_request = {
            "dataset_name": "unit-api",
            "run_name": "unit-api-run",
            "top_k": 1,
            "ground_truth": [
                {
                    "query_id": "ranking",
                    "query": "retrieval precision recall ranking",
                    "relevant_document_ids": [ranking_document_id],
                    "metadata_filter": {"domain": "evaluation"},
                }
            ],
        }
        eval_response = client.post("/evaluate/retrieval", json=eval_request)
        assert eval_response.status_code == 200
        assert eval_response.json()["precision_at_k"] == 1.0

        evaluation_traces = [
            trace
            for trace in trace_repository.list_all()
            if trace.request_payload.get("endpoint") == "/evaluate/retrieval"
        ]
        assert len(evaluation_traces) == 1
        assert evaluation_traces[0].status == "completed"
        evaluation_body = evaluation_traces[0].request_payload["body"]
        assert evaluation_body["dataset_name"] == "unit-api"
        assert evaluation_body["run_name"] == "unit-api-run"
        assert evaluation_body["top_k"] == 1
        assert evaluation_body["strategy"] == "cosine"
        assert evaluation_body["run_benchmark_grid"] is False
        assert evaluation_body["ground_truth"][0]["query_id"] == "ranking"
        assert evaluation_body["ground_truth"][0]["relevant_document_ids"] == [
            ranking_document_id
        ]
        assert evaluation_traces[0].request_payload["retrieval_query"]["query_id"] == "ranking"
        assert evaluation_traces[0].retrieved_chunks[0].document_id == ranking_document_id
        assert evaluation_traces[0].similarity_scores
        assert evaluation_traces[0].retrieval_latency_ms >= 0

        leaderboard_response = client.get("/leaderboard")
        assert leaderboard_response.status_code == 200
        assert leaderboard_response.json()["entries"][0]["name"] == "unit-api-run"
    finally:
        app.dependency_overrides.clear()


def test_retrieval_evaluation_api_records_failed_trace() -> None:
    get_document_repository().clear()
    trace_repository = InMemoryTraceRepository()
    app.dependency_overrides[get_trace_repository] = lambda: trace_repository
    client = TestClient(app)

    try:
        response = client.post(
            "/evaluate/retrieval",
            json={
                "dataset_name": "unit-api",
                "run_name": "missing-relevance",
                "top_k": 1,
                "ground_truth": [
                    {
                        "query_id": "missing-relevance",
                        "query": "retrieval precision recall ranking",
                    }
                ],
            },
        )
        assert response.status_code == 400

        traces = trace_repository.list_all()
        assert len(traces) == 1
        trace = traces[0]
        assert trace.status == "failed"
        assert trace.request_payload["endpoint"] == "/evaluate/retrieval"
        assert trace.request_payload["body"]["run_name"] == "missing-relevance"
        assert trace.retrieved_chunks == []
        assert trace.similarity_scores == []
        assert trace.retrieval_latency_ms >= 0
        assert trace.error_message == "Ground-truth query missing-relevance has no relevant ids."
    finally:
        app.dependency_overrides.clear()


def test_benchmark_suite_writes_report_and_leaderboard(tmp_path) -> None:
    leaderboard_repository = InMemoryLeaderboardRepository()
    suite = RetrievalBenchmarkSuite(leaderboard_repository=leaderboard_repository)
    config = BenchmarkConfig(
        name="unit-benchmark",
        chunk_size_tokens=48,
        chunk_overlap_tokens=6,
        embedding_model="local-hash-32d",
        embedding_dimensions=32,
        retrieval_strategy="keyword_boosted",
        top_k=3,
    )
    report_path = tmp_path / "retrieval_benchmark.md"

    results = suite.run(configs=[config], report_path=report_path)

    assert len(results) == 1
    assert 0 <= results[0].evaluation.recall_at_k <= 1
    assert 0 <= results[0].evaluation.ndcg <= 1
    assert report_path.exists()
    assert "Retrieval Benchmark" in report_path.read_text(encoding="utf-8")
    entries = leaderboard_repository.list_entries()
    assert entries[0].name == "unit-benchmark"
    assert entries[0].report_path == str(report_path)
