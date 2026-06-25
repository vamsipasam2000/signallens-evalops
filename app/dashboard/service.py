from __future__ import annotations

from math import ceil

from app.dashboard.models import (
    DashboardExperiment,
    DashboardFilters,
    DashboardLeaderboardEntry,
    DashboardSortField,
    DashboardSortOrder,
    DashboardSummary,
)
from app.dashboard.repositories import DashboardRepository
from app.quality.models import QualityGateStatus
from app.retrieval.models import LeaderboardEntry


class DashboardService:
    def __init__(self, *, repository: DashboardRepository) -> None:
        self._repository = repository

    def summary(self) -> DashboardSummary:
        traces = self._repository.list_traces()
        quality_gates = self._repository.list_quality_gates()
        total_traces = len(traces)
        successful_requests = sum(1 for trace in traces if trace.status == "completed")
        failed_requests = sum(1 for trace in traces if trace.status == "failed")
        latencies = [trace.retrieval_latency_ms for trace in traces]

        return DashboardSummary(
            total_traces=total_traces,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            success_rate=_ratio(successful_requests, total_traces),
            error_rate=_ratio(failed_requests, total_traces),
            avg_retrieval_latency_ms=_mean(latencies),
            p95_retrieval_latency_ms=_p95(latencies),
            failed_quality_gates=sum(1 for gate in quality_gates if gate.status == "FAILED"),
            total_quality_checks=len(quality_gates),
        )

    def leaderboard(
        self,
        *,
        limit: int = 20,
        sort_by: DashboardSortField = "ndcg",
        sort_order: DashboardSortOrder = "desc",
        filters: DashboardFilters | None = None,
    ) -> list[DashboardLeaderboardEntry]:
        entries = self._filtered_entries(filters=filters)
        quality_by_experiment = self._latest_quality_status_by_experiment()
        ranked = sorted(
            entries,
            key=lambda entry: _sort_value(entry=entry, sort_by=sort_by),
            reverse=sort_order == "desc",
        )
        return [
            _leaderboard_response_entry(
                entry,
                quality_gate_status=quality_by_experiment.get(entry.run_id),
            )
            for entry in ranked[:limit]
        ]

    def experiments(
        self,
        *,
        limit: int = 100,
        filters: DashboardFilters | None = None,
    ) -> list[DashboardExperiment]:
        quality_by_experiment = self._latest_quality_status_by_experiment()
        entries = self._filtered_entries(filters=filters)
        ranked = sorted(
            entries,
            key=lambda entry: (entry.ndcg, entry.mrr, entry.recall_at_k, entry.precision_at_k),
            reverse=True,
        )
        return [
            DashboardExperiment(
                experiment_id=entry.run_id,
                embedding_model=entry.embedding_model,
                chunk_size=entry.chunk_size_tokens,
                retriever=entry.backend,
                top_k=entry.top_k,
                precision=entry.precision_at_k,
                recall=entry.recall_at_k,
                mrr=entry.mrr,
                ndcg=entry.ndcg,
                quality_gate_status=quality_by_experiment.get(entry.run_id),
            )
            for entry in ranked[:limit]
        ]

    def _filtered_entries(
        self,
        *,
        filters: DashboardFilters | None,
    ) -> list[LeaderboardEntry]:
        entries = self._repository.list_leaderboard_entries()
        quality_by_experiment = self._latest_quality_status_by_experiment()
        if filters is None:
            return entries
        return [
            entry
            for entry in entries
            if _matches_filters(
                entry=entry,
                filters=filters,
                quality_gate_status=quality_by_experiment.get(entry.run_id),
            )
        ]

    def _latest_quality_status_by_experiment(self) -> dict[str, QualityGateStatus]:
        status_by_experiment: dict[str, QualityGateStatus] = {}
        for gate in self._repository.list_quality_gates():
            status_by_experiment.setdefault(gate.experiment_id, gate.status)
        return status_by_experiment


def _leaderboard_response_entry(
    entry: LeaderboardEntry,
    *,
    quality_gate_status: QualityGateStatus | None,
) -> DashboardLeaderboardEntry:
    return DashboardLeaderboardEntry(
        experiment_id=entry.run_id,
        name=entry.name,
        dataset_name=entry.dataset_name,
        embedding_model=entry.embedding_model,
        retriever=entry.backend,
        strategy=entry.retrieval_strategy,
        top_k=entry.top_k,
        chunk_size=entry.chunk_size_tokens,
        chunk_overlap=entry.chunk_overlap_tokens,
        precision_at_k=entry.precision_at_k,
        recall_at_k=entry.recall_at_k,
        mrr=entry.mrr,
        ndcg=entry.ndcg,
        avg_retrieval_latency_ms=entry.avg_latency_ms,
        quality_gate_status=quality_gate_status,
    )


def _matches_filters(
    *,
    entry: LeaderboardEntry,
    filters: DashboardFilters,
    quality_gate_status: QualityGateStatus | None,
) -> bool:
    if filters.experiment_id is not None and entry.run_id != filters.experiment_id:
        return False
    if filters.embedding_model is not None and entry.embedding_model != filters.embedding_model:
        return False
    if filters.retriever is not None and entry.backend != filters.retriever:
        return False
    if (
        filters.quality_gate_status is not None
        and quality_gate_status != filters.quality_gate_status
    ):
        return False
    return True


def _sort_value(*, entry: LeaderboardEntry, sort_by: DashboardSortField) -> float:
    if sort_by == "precision_at_k":
        return entry.precision_at_k
    if sort_by == "recall_at_k":
        return entry.recall_at_k
    if sort_by == "mrr":
        return entry.mrr
    if sort_by == "ndcg":
        return entry.ndcg
    if sort_by == "avg_retrieval_latency_ms":
        return entry.avg_latency_ms
    raise ValueError(f"Unsupported dashboard sort field: {sort_by}")


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(ceil(0.95 * len(sorted_values)) - 1, 0)
    return round(sorted_values[index], 3)
