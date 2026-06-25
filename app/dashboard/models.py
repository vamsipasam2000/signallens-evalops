from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.quality.models import QualityGateStatus

DashboardSortField = Literal[
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg",
    "avg_retrieval_latency_ms",
]
DashboardSortOrder = Literal["asc", "desc"]


@dataclass(frozen=True, slots=True)
class DashboardFilters:
    experiment_id: str | None = None
    embedding_model: str | None = None
    retriever: str | None = None
    quality_gate_status: QualityGateStatus | None = None


@dataclass(frozen=True, slots=True)
class DashboardSummary:
    total_traces: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    error_rate: float
    avg_retrieval_latency_ms: float
    p95_retrieval_latency_ms: float
    failed_quality_gates: int
    total_quality_checks: int


@dataclass(frozen=True, slots=True)
class DashboardLeaderboardEntry:
    experiment_id: str
    name: str
    dataset_name: str
    embedding_model: str
    retriever: str
    strategy: str
    top_k: int
    chunk_size: int | None
    chunk_overlap: int | None
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float
    avg_retrieval_latency_ms: float
    quality_gate_status: QualityGateStatus | None


@dataclass(frozen=True, slots=True)
class DashboardExperiment:
    experiment_id: str
    embedding_model: str
    chunk_size: int | None
    retriever: str
    top_k: int
    precision: float
    recall: float
    mrr: float
    ndcg: float
    quality_gate_status: QualityGateStatus | None
