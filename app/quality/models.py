from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from app.analysis.models import FailureType

QualityGateStatus = Literal["PASSED", "FAILED"]
QualityMetricName = Literal[
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg",
    "retrieval_latency_ms",
    "similarity_score",
]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    precision_at_k: float = 0.8
    recall_at_k: float = 0.8
    mrr: float = 0.8
    ndcg: float = 0.8
    retrieval_latency_ms: float = 1_000.0
    similarity_score: float = 0.5


@dataclass(frozen=True, slots=True)
class QualityMetricsSnapshot:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float
    retrieval_latency_ms: float
    similarity_score: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QualityGateFailedCheck:
    metric: QualityMetricName
    actual: float
    required: float
    reason: FailureType
    recommendation: str


@dataclass(frozen=True, slots=True)
class QualityGate:
    experiment_id: str
    status: QualityGateStatus
    failed_checks: list[QualityGateFailedCheck]
    metrics_snapshot: dict[str, float]
    gate_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=utc_now)
