from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

FailureType = Literal[
    "LOW_PRECISION",
    "LOW_RECALL",
    "LOW_MRR",
    "LOW_NDCG",
    "HIGH_LATENCY",
    "LOW_SIMILARITY",
]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class FailureAnalysis:
    query: str
    experiment_id: str
    failure_type: FailureType
    metric_name: str
    metric_value: float
    threshold: float
    recommendation: str
    analysis_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class FailureThresholds:
    precision_at_k: float = 0.7
    recall_at_k: float = 0.7
    mrr: float = 0.7
    ndcg: float = 0.7
    latency_ms: float = 1_000.0
    avg_similarity_score: float = 0.5
