from __future__ import annotations

from typing import Protocol

from app.quality.models import QualityGate
from app.quality.repositories import QualityGateRepository
from app.retrieval.leaderboard import LeaderboardRepository
from app.retrieval.models import LeaderboardEntry
from app.traces.models import Trace
from app.traces.repositories import TraceRepository


class DashboardRepository(Protocol):
    def list_traces(self) -> list[Trace]:
        ...

    def list_leaderboard_entries(self, *, limit: int = 1_000) -> list[LeaderboardEntry]:
        ...

    def list_quality_gates(self, *, limit: int = 1_000) -> list[QualityGate]:
        ...


class CompositeDashboardRepository:
    def __init__(
        self,
        *,
        trace_repository: TraceRepository,
        leaderboard_repository: LeaderboardRepository,
        quality_gate_repository: QualityGateRepository,
    ) -> None:
        self._trace_repository = trace_repository
        self._leaderboard_repository = leaderboard_repository
        self._quality_gate_repository = quality_gate_repository

    def list_traces(self) -> list[Trace]:
        return self._trace_repository.list_all()

    def list_leaderboard_entries(self, *, limit: int = 1_000) -> list[LeaderboardEntry]:
        return self._leaderboard_repository.list_entries(limit=limit)

    def list_quality_gates(self, *, limit: int = 1_000) -> list[QualityGate]:
        return self._quality_gate_repository.list_checks(limit=limit)
