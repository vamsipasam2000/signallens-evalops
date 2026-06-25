from __future__ import annotations

from typing import Literal

from app.analysis.models import FailureAnalysis, FailureThresholds, FailureType
from app.analysis.repositories import FailureAnalysisRepository
from app.retrieval.models import GroundTruthQuery, RetrievalEvaluationSummary

MetricFailureDirection = Literal["below", "above"]

RECOMMENDATIONS: dict[FailureType, str] = {
    "LOW_PRECISION": "Reduce chunk size; improve embeddings; apply metadata filtering.",
    "LOW_RECALL": "Increase top-k; improve chunk overlap.",
    "LOW_MRR": (
        "Improve first-stage retrieval signals; tune ranking so relevant chunks appear earlier."
    ),
    "LOW_NDCG": "Improve ranking strategy; evaluate embedding model.",
    "HIGH_LATENCY": "Reduce chunk count; optimize retrieval backend.",
    "LOW_SIMILARITY": (
        "Improve embeddings; normalize query and document text; validate chunking and filters."
    ),
}


class ErrorAnalysisService:
    def __init__(
        self,
        *,
        repository: FailureAnalysisRepository,
        thresholds: FailureThresholds | None = None,
    ) -> None:
        self._repository = repository
        self._thresholds = thresholds or FailureThresholds()

    def analyze_retrieval_evaluation(
        self,
        *,
        summary: RetrievalEvaluationSummary,
        records: list[GroundTruthQuery] | None = None,
    ) -> list[FailureAnalysis]:
        query_by_id = {record.query_id: record.query for record in records or []}
        analyses: list[FailureAnalysis] = []

        for metric in summary.per_query:
            query = query_by_id.get(metric.query_id, metric.query_id)
            analyses.extend(
                self._failures_for_metric(
                    query=query,
                    experiment_id=summary.run_id,
                    metric_name="precision_at_k",
                    metric_value=metric.precision_at_k,
                    threshold=self._thresholds.precision_at_k,
                    failure_type="LOW_PRECISION",
                    fails_when="below",
                )
            )
            analyses.extend(
                self._failures_for_metric(
                    query=query,
                    experiment_id=summary.run_id,
                    metric_name="recall_at_k",
                    metric_value=metric.recall_at_k,
                    threshold=self._thresholds.recall_at_k,
                    failure_type="LOW_RECALL",
                    fails_when="below",
                )
            )
            analyses.extend(
                self._failures_for_metric(
                    query=query,
                    experiment_id=summary.run_id,
                    metric_name="mrr",
                    metric_value=metric.mrr,
                    threshold=self._thresholds.mrr,
                    failure_type="LOW_MRR",
                    fails_when="below",
                )
            )
            analyses.extend(
                self._failures_for_metric(
                    query=query,
                    experiment_id=summary.run_id,
                    metric_name="ndcg",
                    metric_value=metric.ndcg,
                    threshold=self._thresholds.ndcg,
                    failure_type="LOW_NDCG",
                    fails_when="below",
                )
            )
            analyses.extend(
                self._failures_for_metric(
                    query=query,
                    experiment_id=summary.run_id,
                    metric_name="latency_ms",
                    metric_value=metric.latency_ms,
                    threshold=self._thresholds.latency_ms,
                    failure_type="HIGH_LATENCY",
                    fails_when="above",
                )
            )
            analyses.extend(
                self._failures_for_metric(
                    query=query,
                    experiment_id=summary.run_id,
                    metric_name="avg_similarity_score",
                    metric_value=metric.avg_similarity_score,
                    threshold=self._thresholds.avg_similarity_score,
                    failure_type="LOW_SIMILARITY",
                    fails_when="below",
                )
            )

        return self._repository.save_many(analyses)

    def list_failures(self, *, limit: int = 100) -> list[FailureAnalysis]:
        return self._repository.list_failures(limit=limit)

    def _failures_for_metric(
        self,
        *,
        query: str,
        experiment_id: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
        failure_type: FailureType,
        fails_when: MetricFailureDirection,
    ) -> list[FailureAnalysis]:
        if not _is_failure(metric_value=metric_value, threshold=threshold, fails_when=fails_when):
            return []
        return [
            FailureAnalysis(
                query=query,
                experiment_id=experiment_id,
                failure_type=failure_type,
                metric_name=metric_name,
                metric_value=metric_value,
                threshold=threshold,
                recommendation=RECOMMENDATIONS[failure_type],
            )
        ]


def _is_failure(
    *,
    metric_value: float,
    threshold: float,
    fails_when: MetricFailureDirection,
) -> bool:
    if fails_when == "below":
        return metric_value < threshold
    if fails_when == "above":
        return metric_value > threshold
    raise ValueError(f"Unsupported failure comparison: {fails_when}")
