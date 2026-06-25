from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

RetrievalStrategy = Literal["cosine", "keyword_boosted"]


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    query: str
    top_k: int = 5
    metadata_filter: dict[str, Any] = field(default_factory=dict)
    strategy: RetrievalStrategy = "cosine"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    metadata: dict[str, Any]
    embedding_model: str
    score: float


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    top_k: int
    strategy: RetrievalStrategy
    backend: str
    embedding_model: str
    latency_ms: float
    results: list[RetrievedChunk]
    trace_id: str | None = None


@dataclass(frozen=True, slots=True)
class GroundTruthQuery:
    query_id: str
    query: str
    relevant_chunk_ids: set[str] = field(default_factory=set)
    relevant_document_ids: set[str] = field(default_factory=set)
    metadata_filter: dict[str, Any] = field(default_factory=dict)
    top_k: int | None = None


@dataclass(frozen=True, slots=True)
class RetrievalQueryMetrics:
    query_id: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float
    latency_ms: float
    retrieved_count: int
    relevant_count: int
    avg_similarity_score: float = 0.0


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationSummary:
    run_id: str
    dataset_size: int
    top_k: int
    strategy: RetrievalStrategy
    backend: str
    embedding_model: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float
    avg_latency_ms: float
    per_query: list[RetrievalQueryMetrics]
    avg_similarity_score: float = 0.0
    report_path: str | None = None


@dataclass(frozen=True, slots=True)
class LeaderboardEntry:
    run_id: str
    name: str
    dataset_name: str
    backend: str
    embedding_model: str
    retrieval_strategy: RetrievalStrategy
    top_k: int
    chunk_size_tokens: int | None
    chunk_overlap_tokens: int | None
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float
    avg_latency_ms: float
    report_path: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    name: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    embedding_model: str
    embedding_dimensions: int
    retrieval_strategy: RetrievalStrategy
    top_k: int
    run_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True, slots=True)
class BenchmarkDocument:
    document_id: str
    name: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    config: BenchmarkConfig
    evaluation: RetrievalEvaluationSummary
    leaderboard_entry: LeaderboardEntry
