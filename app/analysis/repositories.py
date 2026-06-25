from __future__ import annotations

from copy import deepcopy
from typing import Protocol

from app.analysis.models import FailureAnalysis


class FailureAnalysisRepository(Protocol):
    def save_many(self, analyses: list[FailureAnalysis]) -> list[FailureAnalysis]:
        ...

    def list_failures(self, *, limit: int = 100) -> list[FailureAnalysis]:
        ...


class InMemoryFailureAnalysisRepository:
    def __init__(self) -> None:
        self._analyses: dict[str, FailureAnalysis] = {}

    def save_many(self, analyses: list[FailureAnalysis]) -> list[FailureAnalysis]:
        for analysis in analyses:
            self._analyses[analysis.analysis_id] = deepcopy(analysis)
        return deepcopy(analyses)

    def list_failures(self, *, limit: int = 100) -> list[FailureAnalysis]:
        return deepcopy(
            sorted(
                self._analyses.values(),
                key=lambda analysis: analysis.timestamp,
                reverse=True,
            )[:limit]
        )

    def clear(self) -> None:
        self._analyses.clear()
