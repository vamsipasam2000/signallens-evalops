from __future__ import annotations

from typing import Protocol

from app.traces.models import Trace
from app.traces.repositories import TraceRepository


class MetricsRepository(Protocol):
    def list_traces(self) -> list[Trace]:
        ...


class TraceBackedMetricsRepository:
    def __init__(self, *, trace_repository: TraceRepository) -> None:
        self._trace_repository = trace_repository

    def list_traces(self) -> list[Trace]:
        return self._trace_repository.list_all()
