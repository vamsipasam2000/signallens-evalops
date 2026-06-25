from typing import Annotated

from fastapi import Depends

from app.analysis.models import FailureThresholds
from app.analysis.registry import get_in_memory_failure_analysis_repository
from app.analysis.repositories import FailureAnalysisRepository
from app.analysis.service import ErrorAnalysisService
from app.core.config import Settings, get_settings
from app.core.errors import DependencyUnavailableError
from app.dashboard.repositories import CompositeDashboardRepository, DashboardRepository
from app.dashboard.service import DashboardService
from app.metrics.repositories import MetricsRepository, TraceBackedMetricsRepository
from app.metrics.service import MetricsService
from app.mlflow.tracking import (
    ExperimentTracker,
    MLflowExperimentTracker,
    NoOpExperimentTracker,
)
from app.quality.models import QualityThresholds
from app.quality.registry import get_in_memory_quality_gate_repository
from app.quality.repositories import QualityGateRepository
from app.quality.service import QualityGateService
from app.rag.chunking import TokenWindowChunker
from app.rag.embeddings import EmbeddingProvider, HashEmbeddingProvider
from app.rag.ingestion import IngestionService
from app.rag.parsers import default_parser_router
from app.retrieval.adapters.chroma import ChromaRetriever
from app.retrieval.adapters.pgvector import PgVectorRetriever
from app.retrieval.benchmark import RetrievalBenchmarkSuite
from app.retrieval.evaluation import RetrievalEvaluationService
from app.retrieval.leaderboard import FileLeaderboardRepository, LeaderboardRepository
from app.retrieval.retrievers import RepositoryRetriever, Retriever, VectorIndex
from app.retrieval.service import RetrievalService
from app.storage.registry import get_document_repository
from app.storage.repositories import DocumentRepository
from app.traces.registry import get_in_memory_trace_repository
from app.traces.repositories import PostgresTraceRepository, TraceRepository
from app.traces.service import TraceService


def get_embedding_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> EmbeddingProvider:
    if settings.embedding_provider == "local-hash":
        return HashEmbeddingProvider(
            dimensions=settings.local_embedding_dimensions,
        )
    if settings.embedding_provider == "sentence-transformers":
        from app.rag.embeddings import SentenceTransformerEmbeddingProvider

        return SentenceTransformerEmbeddingProvider()
    raise DependencyUnavailableError(
        f"Unsupported embedding provider: {settings.embedding_provider}"
    )


def get_vector_index(
    settings: Annotated[Settings, Depends(get_settings)],
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
) -> VectorIndex | None:
    if settings.vector_backend == "memory":
        return None
    if settings.vector_backend == "chroma":
        return ChromaRetriever(
            persist_directory=settings.chroma_persist_directory,
            collection_name=settings.chroma_collection_name,
            embedding_provider=embedding_provider,
        )
    if settings.vector_backend == "pgvector":
        if not settings.pgvector_dsn:
            raise DependencyUnavailableError("PGVECTOR_DSN is required for pgvector backend.")
        return PgVectorRetriever(
            dsn=settings.pgvector_dsn,
            table_name=settings.pgvector_table_name,
            embedding_dimensions=embedding_provider.dimensions,
            embedding_provider=embedding_provider,
        )
    raise DependencyUnavailableError(f"Unsupported vector backend: {settings.vector_backend}")


def get_rag_ingestion_service(
    settings: Annotated[Settings, Depends(get_settings)],
    repository: Annotated[DocumentRepository, Depends(get_document_repository)],
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    vector_index: Annotated[VectorIndex | None, Depends(get_vector_index)],
) -> IngestionService:
    return IngestionService(
        parser_router=default_parser_router(),
        chunker_factory=lambda chunk_size, overlap: TokenWindowChunker(
            chunk_size_tokens=chunk_size,
            overlap_tokens=overlap,
        ),
        embedding_provider=embedding_provider,
        repository=repository,
        default_chunk_size_tokens=settings.default_chunk_size_tokens,
        default_chunk_overlap_tokens=settings.default_chunk_overlap_tokens,
        vector_index=vector_index,
    )


def get_retriever(
    settings: Annotated[Settings, Depends(get_settings)],
    repository: Annotated[DocumentRepository, Depends(get_document_repository)],
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
) -> Retriever:
    if settings.vector_backend == "memory":
        return RepositoryRetriever(
            repository=repository,
            embedding_provider=embedding_provider,
        )
    if settings.vector_backend == "chroma":
        return ChromaRetriever(
            persist_directory=settings.chroma_persist_directory,
            collection_name=settings.chroma_collection_name,
            embedding_provider=embedding_provider,
        )
    if settings.vector_backend == "pgvector":
        if not settings.pgvector_dsn:
            raise DependencyUnavailableError("PGVECTOR_DSN is required for pgvector backend.")
        return PgVectorRetriever(
            dsn=settings.pgvector_dsn,
            table_name=settings.pgvector_table_name,
            embedding_dimensions=embedding_provider.dimensions,
            embedding_provider=embedding_provider,
        )
    raise DependencyUnavailableError(f"Unsupported vector backend: {settings.vector_backend}")


def get_trace_repository(
    settings: Annotated[Settings, Depends(get_settings)],
) -> TraceRepository:
    if settings.trace_storage_backend == "memory":
        return get_in_memory_trace_repository()
    if settings.trace_storage_backend == "postgres":
        if not settings.trace_postgres_dsn:
            raise DependencyUnavailableError(
                "TRACE_POSTGRES_DSN is required for postgres trace storage."
            )
        return PostgresTraceRepository(
            dsn=settings.trace_postgres_dsn,
            table_name=settings.trace_postgres_table_name,
        )
    raise DependencyUnavailableError(
        f"Unsupported trace storage backend: {settings.trace_storage_backend}"
    )


def get_retrieval_service(
    retriever: Annotated[Retriever, Depends(get_retriever)],
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    trace_repository: Annotated[TraceRepository, Depends(get_trace_repository)],
) -> RetrievalService:
    return RetrievalService(
        retriever=retriever,
        trace_repository=trace_repository,
        embedding_model=embedding_provider.model_name,
    )


def get_retrieval_evaluation_service(
    retrieval_service: Annotated[RetrievalService, Depends(get_retrieval_service)],
    retriever: Annotated[Retriever, Depends(get_retriever)],
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
) -> RetrievalEvaluationService:
    return RetrievalEvaluationService(
        retrieval_service=retrieval_service,
        backend=retriever.backend_name,
        embedding_model=embedding_provider.model_name,
    )


def get_leaderboard_repository(
    settings: Annotated[Settings, Depends(get_settings)],
) -> LeaderboardRepository:
    from pathlib import Path

    return FileLeaderboardRepository(path=Path(settings.retrieval_leaderboard_path))


def get_experiment_tracker(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ExperimentTracker:
    if not settings.mlflow_enabled:
        return NoOpExperimentTracker()
    return MLflowExperimentTracker(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.mlflow_experiment_name,
    )


def get_retrieval_benchmark_suite(
    leaderboard_repository: Annotated[
        LeaderboardRepository,
        Depends(get_leaderboard_repository),
    ],
    tracker: Annotated[ExperimentTracker, Depends(get_experiment_tracker)],
    trace_repository: Annotated[
        TraceRepository,
        Depends(get_trace_repository),
    ],
) -> RetrievalBenchmarkSuite:
    return RetrievalBenchmarkSuite(
        leaderboard_repository=leaderboard_repository,
        tracker=tracker,
        trace_repository=trace_repository,
    )


def get_trace_service(
    trace_repository: Annotated[TraceRepository, Depends(get_trace_repository)],
) -> TraceService:
    return TraceService(repository=trace_repository)


def get_metrics_repository(
    trace_repository: Annotated[TraceRepository, Depends(get_trace_repository)],
) -> MetricsRepository:
    return TraceBackedMetricsRepository(trace_repository=trace_repository)


def get_metrics_service(
    metrics_repository: Annotated[MetricsRepository, Depends(get_metrics_repository)],
) -> MetricsService:
    return MetricsService(repository=metrics_repository)


def get_failure_analysis_repository() -> FailureAnalysisRepository:
    return get_in_memory_failure_analysis_repository()


def get_error_analysis_service(
    settings: Annotated[Settings, Depends(get_settings)],
    repository: Annotated[
        FailureAnalysisRepository,
        Depends(get_failure_analysis_repository),
    ],
) -> ErrorAnalysisService:
    return ErrorAnalysisService(
        repository=repository,
        thresholds=FailureThresholds(
            precision_at_k=settings.analysis_precision_threshold,
            recall_at_k=settings.analysis_recall_threshold,
            mrr=settings.analysis_mrr_threshold,
            ndcg=settings.analysis_ndcg_threshold,
            latency_ms=settings.analysis_latency_threshold_ms,
            avg_similarity_score=settings.analysis_similarity_threshold,
        ),
    )


def get_quality_gate_repository() -> QualityGateRepository:
    return get_in_memory_quality_gate_repository()


def get_quality_gate_service(
    settings: Annotated[Settings, Depends(get_settings)],
    repository: Annotated[
        QualityGateRepository,
        Depends(get_quality_gate_repository),
    ],
) -> QualityGateService:
    return QualityGateService(
        repository=repository,
        thresholds=QualityThresholds(
            precision_at_k=settings.quality_precision_threshold,
            recall_at_k=settings.quality_recall_threshold,
            mrr=settings.quality_mrr_threshold,
            ndcg=settings.quality_ndcg_threshold,
            retrieval_latency_ms=settings.quality_latency_threshold_ms,
            similarity_score=settings.quality_similarity_threshold,
        ),
    )


def get_dashboard_repository(
    trace_repository: Annotated[TraceRepository, Depends(get_trace_repository)],
    leaderboard_repository: Annotated[
        LeaderboardRepository,
        Depends(get_leaderboard_repository),
    ],
    quality_gate_repository: Annotated[
        QualityGateRepository,
        Depends(get_quality_gate_repository),
    ],
) -> DashboardRepository:
    return CompositeDashboardRepository(
        trace_repository=trace_repository,
        leaderboard_repository=leaderboard_repository,
        quality_gate_repository=quality_gate_repository,
    )


def get_dashboard_service(
    repository: Annotated[DashboardRepository, Depends(get_dashboard_repository)],
) -> DashboardService:
    return DashboardService(repository=repository)
