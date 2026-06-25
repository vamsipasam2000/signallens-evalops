from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from app.traces.models import Trace, TraceSortField, TraceSortOrder, TraceStatus, TraceSummary
from app.traces.repositories import TraceRepository


@dataclass(frozen=True, slots=True)
class TraceListFilters:
    status: TraceStatus | None = None
    retriever_name: str | None = None
    embedding_model: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None


@dataclass(frozen=True, slots=True)
class TraceListResult:
    items: list[TraceSummary]
    total: int
    limit: int
    offset: int
    sort_by: TraceSortField
    sort_order: TraceSortOrder


class TraceService:
    def __init__(self, *, repository: TraceRepository) -> None:
        self._repository = repository

    def record(self, trace: Trace) -> Trace:
        return self._repository.save(trace)

    def get(self, trace_id: str) -> Trace | None:
        return self._repository.get(trace_id)

    def list(
        self,
        *,
        filters: TraceListFilters,
        limit: int,
        offset: int,
        sort_by: TraceSortField,
        sort_order: TraceSortOrder,
    ) -> TraceListResult:
        traces = [
            trace
            for trace in self._repository.list_all()
            if _matches_filters(trace=trace, filters=filters)
        ]
        descending = sort_order == "desc"
        sorted_traces = sorted(
            traces,
            key=lambda trace: getattr(trace, sort_by),
            reverse=descending,
        )
        page = sorted_traces[offset : offset + limit]
        return TraceListResult(
            items=[_to_summary(trace) for trace in page],
            total=len(sorted_traces),
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order,
        )


def _matches_filters(*, trace: Trace, filters: TraceListFilters) -> bool:
    if filters.status is not None and trace.status != filters.status:
        return False
    if filters.retriever_name is not None and trace.retriever_name != filters.retriever_name:
        return False
    if filters.embedding_model is not None and trace.embedding_model != filters.embedding_model:
        return False
    if filters.start_date is not None and trace.timestamp < _normalize_datetime(filters.start_date):
        return False
    if filters.end_date is not None and trace.timestamp > _normalize_datetime(filters.end_date):
        return False
    return True


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _to_summary(trace: Trace) -> TraceSummary:
    return TraceSummary(
        trace_id=trace.trace_id,
        timestamp=trace.timestamp,
        query=trace.query,
        retriever_name=trace.retriever_name,
        embedding_model=trace.embedding_model,
        status=trace.status,
        retrieval_latency_ms=trace.retrieval_latency_ms,
        total_latency_ms=trace.total_latency_ms,
    )
