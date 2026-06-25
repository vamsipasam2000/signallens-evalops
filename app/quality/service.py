from __future__ import annotations

from typing import Literal

from app.analysis.models import FailureType
from app.quality.models import (
    QualityGate,
    QualityGateFailedCheck,
    QualityMetricName,
    QualityMetricsSnapshot,
    QualityThresholds,
)
from app.quality.repositories import QualityGateRepository
from app.retrieval.models import RetrievalEvaluationSummary

QualityComparison = Literal["below", "above"]

QUALITY_RECOMMENDATIONS: dict[FailureType, str] = {
    "LOW_PRECISION": (
        "Review chunking strategy; evaluate embedding model; apply metadata filtering."
    ),
    "LOW_RECALL": "Increase top-k; improve chunk overlap; expand retrieval filters.",
    "LOW_MRR": "Tune first-stage retrieval and reranking so relevant chunks appear earlier.",
    "LOW_NDCG": "Improve ranking strategy; evaluate embedding model.",
    "HIGH_LATENCY": "Reduce chunk count; optimize retrieval backend; tune indexes.",
    "LOW_SIMILARITY": "Improve embeddings; validate query normalization and chunk quality.",
}


class QualityGateService:
    def __init__(
        self,
        *,
        repository: QualityGateRepository,
        thresholds: QualityThresholds | None = None,
    ) -> None:
        self._repository = repository
        self._thresholds = thresholds or QualityThresholds()

    def evaluate_experiment(
        self,
        *,
        experiment_id: str,
        metrics_snapshot: QualityMetricsSnapshot,
    ) -> QualityGate:
        failed_checks = self._failed_checks(metrics_snapshot)
        gate = QualityGate(
            experiment_id=experiment_id,
            status="FAILED" if failed_checks else "PASSED",
            failed_checks=failed_checks,
            metrics_snapshot=metrics_snapshot.to_dict(),
        )
        return self._repository.save(gate)

    def evaluate_retrieval_summary(self, summary: RetrievalEvaluationSummary) -> QualityGate:
        return self.evaluate_experiment(
            experiment_id=summary.run_id,
            metrics_snapshot=QualityMetricsSnapshot(
                precision_at_k=summary.precision_at_k,
                recall_at_k=summary.recall_at_k,
                mrr=summary.mrr,
                ndcg=summary.ndcg,
                retrieval_latency_ms=summary.avg_latency_ms,
                similarity_score=summary.avg_similarity_score,
            ),
        )

    def list_checks(self, *, limit: int = 100) -> list[QualityGate]:
        return self._repository.list_checks(limit=limit)

    def _failed_checks(
        self,
        metrics_snapshot: QualityMetricsSnapshot,
    ) -> list[QualityGateFailedCheck]:
        return [
            *self._check_metric(
                metric="precision_at_k",
                actual=metrics_snapshot.precision_at_k,
                required=self._thresholds.precision_at_k,
                reason="LOW_PRECISION",
                fails_when="below",
            ),
            *self._check_metric(
                metric="recall_at_k",
                actual=metrics_snapshot.recall_at_k,
                required=self._thresholds.recall_at_k,
                reason="LOW_RECALL",
                fails_when="below",
            ),
            *self._check_metric(
                metric="mrr",
                actual=metrics_snapshot.mrr,
                required=self._thresholds.mrr,
                reason="LOW_MRR",
                fails_when="below",
            ),
            *self._check_metric(
                metric="ndcg",
                actual=metrics_snapshot.ndcg,
                required=self._thresholds.ndcg,
                reason="LOW_NDCG",
                fails_when="below",
            ),
            *self._check_metric(
                metric="retrieval_latency_ms",
                actual=metrics_snapshot.retrieval_latency_ms,
                required=self._thresholds.retrieval_latency_ms,
                reason="HIGH_LATENCY",
                fails_when="above",
            ),
            *self._check_metric(
                metric="similarity_score",
                actual=metrics_snapshot.similarity_score,
                required=self._thresholds.similarity_score,
                reason="LOW_SIMILARITY",
                fails_when="below",
            ),
        ]

    def _check_metric(
        self,
        *,
        metric: QualityMetricName,
        actual: float,
        required: float,
        reason: FailureType,
        fails_when: QualityComparison,
    ) -> list[QualityGateFailedCheck]:
        if not _is_failed(actual=actual, required=required, fails_when=fails_when):
            return []
        return [
            QualityGateFailedCheck(
                metric=metric,
                actual=actual,
                required=required,
                reason=reason,
                recommendation=QUALITY_RECOMMENDATIONS[reason],
            )
        ]


def _is_failed(*, actual: float, required: float, fails_when: QualityComparison) -> bool:
    if fails_when == "below":
        return actual < required
    if fails_when == "above":
        return actual > required
    raise ValueError(f"Unsupported quality gate comparison: {fails_when}")
