from __future__ import annotations

from copy import deepcopy
from typing import Protocol

from app.quality.models import QualityGate


class QualityGateRepository(Protocol):
    def save(self, gate: QualityGate) -> QualityGate:
        ...

    def list_checks(self, *, limit: int = 100) -> list[QualityGate]:
        ...


class InMemoryQualityGateRepository:
    def __init__(self) -> None:
        self._checks: dict[str, QualityGate] = {}

    def save(self, gate: QualityGate) -> QualityGate:
        self._checks[gate.gate_id] = deepcopy(gate)
        return deepcopy(gate)

    def list_checks(self, *, limit: int = 100) -> list[QualityGate]:
        return deepcopy(
            sorted(
                self._checks.values(),
                key=lambda gate: gate.timestamp,
                reverse=True,
            )[:limit]
        )

    def clear(self) -> None:
        self._checks.clear()
