from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.mlflow.tracking import ExperimentTracker, NoOpExperimentTracker
from app.rag.chunking import TokenWindowChunker
from app.rag.embeddings import HashEmbeddingProvider
from app.retrieval.evaluation import RetrievalEvaluationService
from app.retrieval.leaderboard import FileLeaderboardRepository, LeaderboardRepository
from app.retrieval.models import (
    BenchmarkConfig,
    BenchmarkDocument,
    BenchmarkRunResult,
    GroundTruthQuery,
    LeaderboardEntry,
    RetrievalEvaluationSummary,
)
from app.retrieval.retrievers import RepositoryRetriever
from app.retrieval.service import RetrievalService
from app.storage.models import ChunkRecord, DocumentRecord
from app.storage.repositories import InMemoryDocumentRepository
from app.traces.repositories import TraceRepository

DEFAULT_REPORT_PATH = Path("reports/retrieval_benchmark_sprint2.md")
DEFAULT_GROUND_TRUTH_PATH = Path(__file__).resolve().parents[1] / "data" / (
    "retrieval_ground_truth.jsonl"
)


DEFAULT_DOCUMENTS: tuple[BenchmarkDocument, ...] = (
    BenchmarkDocument(
        document_id="doc-ranking-metrics",
        name="ranking-metrics.md",
        metadata={"domain": "evaluation", "topic": "ranking"},
        text=(
            "Retrieval quality is measured with Precision at K, Recall at K, "
            "mean reciprocal rank, and NDCG. Precision at K rewards returning "
            "relevant chunks in the visible results. Recall at K checks whether "
            "the known relevant context was found. MRR rewards the first relevant "
            "chunk appearing near the top. NDCG rewards ranking all useful chunks "
            "early in the result list for search evaluation."
        ),
    ),
    BenchmarkDocument(
        document_id="doc-chromadb",
        name="chromadb-adapter.md",
        metadata={"domain": "storage", "backend": "chroma"},
        text=(
            "ChromaDB provides persistent local vector storage through named "
            "collections. A retrieval platform uses Chroma collections to upsert "
            "chunk embeddings, store metadata, and run cosine similarity search "
            "against query embeddings. Collection management keeps development "
            "and benchmark indexes isolated."
        ),
    ),
    BenchmarkDocument(
        document_id="doc-pgvector",
        name="pgvector-adapter.md",
        metadata={"domain": "storage", "backend": "pgvector"},
        text=(
            "pgvector integrates vector search with PostgreSQL. Production systems "
            "create the vector extension, store embeddings in vector columns, add "
            "HNSW or IVFFlat indexes with cosine operators, and combine metadata "
            "filters with similarity search inside SQL."
        ),
    ),
    BenchmarkDocument(
        document_id="doc-mlflow",
        name="mlflow-experiments.md",
        metadata={"domain": "observability", "topic": "mlflow"},
        text=(
            "MLflow experiment tracking records retrieval parameters, metric values, "
            "and report artifacts. Retrieval teams compare embedding models, chunk "
            "sizes, top K settings, and ranking strategies by logging each benchmark "
            "run into the same experiment."
        ),
    ),
    BenchmarkDocument(
        document_id="doc-ingestion",
        name="rag-ingestion.md",
        metadata={"domain": "ingestion", "topic": "chunking"},
        text=(
            "RAG ingestion parses documents, splits text into token windows, applies "
            "chunk overlap, embeds each chunk, and stores metadata. Chunk size changes "
            "retrieval behavior because small chunks improve specificity while larger "
            "chunks preserve context."
        ),
    ),
)


class RetrievalBenchmarkSuite:
    def __init__(
        self,
        *,
        leaderboard_repository: LeaderboardRepository,
        tracker: ExperimentTracker | None = None,
        trace_repository: TraceRepository | None = None,
    ) -> None:
        self._leaderboard_repository = leaderboard_repository
        self._tracker = tracker or NoOpExperimentTracker()
        self._trace_repository = trace_repository

    def run(
        self,
        *,
        configs: list[BenchmarkConfig] | None = None,
        documents: tuple[BenchmarkDocument, ...] = DEFAULT_DOCUMENTS,
        ground_truth: list[GroundTruthQuery] | None = None,
        dataset_name: str = "default-retrieval-ground-truth",
        report_path: Path = DEFAULT_REPORT_PATH,
        trace_request_payload: dict[str, Any] | None = None,
    ) -> list[BenchmarkRunResult]:
        effective_configs = configs or default_benchmark_grid()
        effective_ground_truth = ground_truth or load_ground_truth_records()
        results: list[BenchmarkRunResult] = []

        for config in effective_configs:
            repository = _build_repository(config=config, documents=documents)
            embedding_provider = HashEmbeddingProvider(
                dimensions=config.embedding_dimensions,
                model_name=config.embedding_model,
            )
            retriever = RepositoryRetriever(
                repository=repository,
                embedding_provider=embedding_provider,
            )
            retrieval_service = RetrievalService(
                retriever=retriever,
                trace_repository=self._trace_repository,
                embedding_model=embedding_provider.model_name,
            )
            evaluator = RetrievalEvaluationService(
                retrieval_service=retrieval_service,
                backend=retriever.backend_name,
                embedding_model=embedding_provider.model_name,
            )
            summary = evaluator.evaluate(
                records=effective_ground_truth,
                top_k=config.top_k,
                strategy=config.retrieval_strategy,
                run_id=config.run_id,
                trace_request_payload=_benchmark_trace_payload(
                    base_payload=trace_request_payload,
                    dataset_name=dataset_name,
                    config=config,
                ),
            )
            entry = evaluator.to_leaderboard_entry(
                summary=summary,
                name=config.name,
                dataset_name=dataset_name,
                chunk_size_tokens=config.chunk_size_tokens,
                chunk_overlap_tokens=config.chunk_overlap_tokens,
                parameters={
                    "embedding_dimensions": config.embedding_dimensions,
                    "retrieval_strategy": config.retrieval_strategy,
                },
            )
            results.append(
                BenchmarkRunResult(
                    config=config,
                    evaluation=summary,
                    leaderboard_entry=entry,
                )
            )

        write_benchmark_report(results=results, path=report_path)
        report_artifact = report_path
        reported_results = [
            BenchmarkRunResult(
                config=result.config,
                evaluation=_with_summary_report_path(result.evaluation, report_path),
                leaderboard_entry=_with_report_path(result.leaderboard_entry, report_path),
            )
            for result in results
        ]
        for result in results:
            entry = _with_report_path(result.leaderboard_entry, report_path)
            self._leaderboard_repository.record(entry)
            self._tracker.log_run(
                run_name=result.config.name,
                parameters={
                    "dataset_name": dataset_name,
                    "backend": entry.backend,
                    "embedding_model": entry.embedding_model,
                    "retrieval_strategy": entry.retrieval_strategy,
                    "top_k": entry.top_k,
                    "chunk_size_tokens": entry.chunk_size_tokens,
                    "chunk_overlap_tokens": entry.chunk_overlap_tokens,
                    **entry.parameters,
                },
                metrics={
                    "precision_at_k": entry.precision_at_k,
                    "recall_at_k": entry.recall_at_k,
                    "mrr": entry.mrr,
                    "ndcg": entry.ndcg,
                    "avg_latency_ms": entry.avg_latency_ms,
                },
                artifacts=[report_artifact],
            )

        return reported_results


def _benchmark_trace_payload(
    *,
    base_payload: dict[str, Any] | None,
    dataset_name: str,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    payload = dict(base_payload or {})
    payload["dataset_name"] = dataset_name
    payload["benchmark_config"] = asdict(config)
    return payload


def default_benchmark_grid() -> list[BenchmarkConfig]:
    configs: list[BenchmarkConfig] = []
    for chunk_size in (24, 48, 96):
        overlap = 6 if chunk_size <= 48 else 12
        for embedding_dimensions in (64, 128):
            for strategy in ("cosine", "keyword_boosted"):
                for top_k in (1, 3, 5):
                    model = f"local-hash-{embedding_dimensions}d"
                    configs.append(
                        BenchmarkConfig(
                            name=(
                                f"chunk-{chunk_size}_embed-{embedding_dimensions}d_"
                                f"{strategy}_top-{top_k}"
                            ),
                            chunk_size_tokens=chunk_size,
                            chunk_overlap_tokens=overlap,
                            embedding_model=model,
                            embedding_dimensions=embedding_dimensions,
                            retrieval_strategy=strategy,
                            top_k=top_k,
                        )
                    )
    return configs


def load_ground_truth_records(path: Path = DEFAULT_GROUND_TRUTH_PATH) -> list[GroundTruthQuery]:
    records: list[GroundTruthQuery] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                records.append(
                    GroundTruthQuery(
                        query_id=payload["query_id"],
                        query=payload["query"],
                        relevant_chunk_ids=set(payload.get("relevant_chunk_ids", [])),
                        relevant_document_ids=set(payload.get("relevant_document_ids", [])),
                        metadata_filter=dict(payload.get("metadata_filter", {})),
                        top_k=payload.get("top_k"),
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"Invalid retrieval ground truth at line {line_number}: missing {exc}"
                ) from exc
    return records


def write_benchmark_report(*, results: list[BenchmarkRunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        results,
        key=lambda result: (
            result.leaderboard_entry.ndcg,
            result.leaderboard_entry.mrr,
            result.leaderboard_entry.recall_at_k,
            result.leaderboard_entry.precision_at_k,
            -result.leaderboard_entry.avg_latency_ms,
        ),
        reverse=True,
    )
    lines = [
        "# SignalLens EvalOps 2.0 Sprint 2 Retrieval Benchmark",
        "",
        "This report is generated by `app.retrieval.benchmark` using the deterministic "
        "offline retrieval corpus and ground-truth query set.",
        "",
        "## Leaderboard",
        "",
        (
            "| Rank | Run | Chunk | Embedding | Strategy | Top-K | Precision@K | "
            "Recall@K | MRR | NDCG | Avg Latency ms |"
        ),
        "|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, result in enumerate(ranked, start=1):
        entry = result.leaderboard_entry
        lines.append(
            "| "
            f"{rank} | {entry.name} | {entry.chunk_size_tokens} | "
            f"{entry.embedding_model} | {entry.retrieval_strategy} | {entry.top_k} | "
            f"{entry.precision_at_k:.4f} | {entry.recall_at_k:.4f} | "
            f"{entry.mrr:.4f} | {entry.ndcg:.4f} | {entry.avg_latency_ms:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Comparison Dimensions",
            "",
            "- Chunk sizes: 24, 48, and 96 token windows with overlap.",
            "- Embedding models: deterministic local hash embeddings at 64d and 128d.",
            "- Retrieval strategies: cosine and keyword-boosted cosine.",
            "- Top-K settings: 1, 3, and 5.",
            "",
            "## Raw Runs",
            "",
            "```json",
            json.dumps(
                [
                    {
                        "config": asdict(result.config),
                        "metrics": asdict(result.leaderboard_entry),
                    }
                    for result in ranked
                ],
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_repository(
    *,
    config: BenchmarkConfig,
    documents: tuple[BenchmarkDocument, ...],
) -> InMemoryDocumentRepository:
    repository = InMemoryDocumentRepository()
    chunker = TokenWindowChunker(
        chunk_size_tokens=config.chunk_size_tokens,
        overlap_tokens=config.chunk_overlap_tokens,
    )
    embedding_provider = HashEmbeddingProvider(
        dimensions=config.embedding_dimensions,
        model_name=config.embedding_model,
    )

    for document in documents:
        chunks = chunker.chunk(
            document.text,
            metadata={
                **document.metadata,
                "document_name": document.name,
            },
        )
        embeddings = embedding_provider.embed_documents([chunk.text for chunk in chunks])
        repository.save_document(
            DocumentRecord(
                document_id=document.document_id,
                name=document.name,
                content_type="text/markdown",
                source="benchmark",
                metadata=document.metadata,
                chunk_count=len(chunks),
                token_count=len(document.text.split()),
                ingestion_latency_ms=0.0,
                parser_version="benchmark-fixture-v1",
                chunker_version=chunker.version,
                embedding_model=embedding_provider.model_name,
            )
        )
        repository.save_chunks(
            [
                ChunkRecord(
                    chunk_id=f"{document.document_id}-chunk-{chunk.chunk_index}",
                    document_id=document.document_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    metadata=chunk.metadata,
                    embedding=embedding,
                    embedding_model=embedding_provider.model_name,
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ]
        )
    return repository


def _with_report_path(entry: LeaderboardEntry, report_path: Path) -> LeaderboardEntry:
    return LeaderboardEntry(
        run_id=entry.run_id,
        name=entry.name,
        dataset_name=entry.dataset_name,
        backend=entry.backend,
        embedding_model=entry.embedding_model,
        retrieval_strategy=entry.retrieval_strategy,
        top_k=entry.top_k,
        chunk_size_tokens=entry.chunk_size_tokens,
        chunk_overlap_tokens=entry.chunk_overlap_tokens,
        precision_at_k=entry.precision_at_k,
        recall_at_k=entry.recall_at_k,
        mrr=entry.mrr,
        ndcg=entry.ndcg,
        avg_latency_ms=entry.avg_latency_ms,
        report_path=str(report_path),
        parameters=entry.parameters,
    )


def _with_summary_report_path(
    summary: RetrievalEvaluationSummary,
    report_path: Path,
) -> RetrievalEvaluationSummary:
    return RetrievalEvaluationSummary(
        run_id=summary.run_id,
        dataset_size=summary.dataset_size,
        top_k=summary.top_k,
        strategy=summary.strategy,
        backend=summary.backend,
        embedding_model=summary.embedding_model,
        precision_at_k=summary.precision_at_k,
        recall_at_k=summary.recall_at_k,
        mrr=summary.mrr,
        ndcg=summary.ndcg,
        avg_latency_ms=summary.avg_latency_ms,
        per_query=summary.per_query,
        avg_similarity_score=summary.avg_similarity_score,
        report_path=str(report_path),
    )


def main() -> None:
    suite = RetrievalBenchmarkSuite(
        leaderboard_repository=FileLeaderboardRepository(
            path=Path("reports/retrieval_leaderboard.json"),
        )
    )
    results = suite.run()
    best = max(
        results,
        key=lambda result: (
            result.evaluation.ndcg,
            result.evaluation.mrr,
            result.evaluation.recall_at_k,
            result.evaluation.precision_at_k,
        ),
    )
    print(
        json.dumps(
            {
                "runs": len(results),
                "report_path": best.evaluation.report_path,
                "best_run": best.config.name,
                "best_metrics": {
                    "precision_at_k": best.evaluation.precision_at_k,
                    "recall_at_k": best.evaluation.recall_at_k,
                    "mrr": best.evaluation.mrr,
                    "ndcg": best.evaluation.ndcg,
                    "avg_latency_ms": best.evaluation.avg_latency_ms,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
