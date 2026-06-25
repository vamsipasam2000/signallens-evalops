from __future__ import annotations

from math import ceil

from app.metrics.models import MetricSnapshot
from app.metrics.repositories import MetricsRepository
from app.traces.models import utc_now


class MetricsService:
    def __init__(self, *, repository: MetricsRepository) -> None:
        self._repository = repository

    def summarize(self) -> MetricSnapshot:
        traces = self._repository.list_traces()
        total_requests = len(traces)
        failed_requests = sum(1 for trace in traces if trace.status == "failed")
        successful_requests = sum(1 for trace in traces if trace.status == "completed")
        retrieval_latencies = [trace.retrieval_latency_ms for trace in traces]
        chunk_counts = [len(trace.retrieved_chunks) for trace in traces]
        similarity_scores = [
            score
            for trace in traces
            for score in trace.similarity_scores
        ]

        return MetricSnapshot(
            timestamp=utc_now(),
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            success_rate=_ratio(successful_requests, total_requests),
            error_rate=_ratio(failed_requests, total_requests),
            avg_retrieval_latency_ms=_mean(retrieval_latencies, digits=3),
            p95_retrieval_latency_ms=_p95(retrieval_latencies),
            avg_chunks_returned=_mean(chunk_counts, digits=3),
            avg_similarity_score=_mean(similarity_scores, digits=4),
            request_volume=total_requests,
            trace_volume=total_requests,
            failure_count=failed_requests,
        )


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _mean(values: list[float] | list[int], *, digits: int) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), digits)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(ceil(0.95 * len(sorted_values)) - 1, 0)
    return round(sorted_values[index], 3)
