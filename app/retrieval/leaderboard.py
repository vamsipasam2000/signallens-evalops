from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from app.retrieval.models import LeaderboardEntry


class LeaderboardRepository(Protocol):
    def record(self, entry: LeaderboardEntry) -> LeaderboardEntry:
        ...

    def list_entries(self, *, limit: int = 20) -> list[LeaderboardEntry]:
        ...


class InMemoryLeaderboardRepository:
    def __init__(self) -> None:
        self._entries: dict[str, LeaderboardEntry] = {}

    def record(self, entry: LeaderboardEntry) -> LeaderboardEntry:
        self._entries[entry.run_id] = entry
        return entry

    def list_entries(self, *, limit: int = 20) -> list[LeaderboardEntry]:
        return _rank_entries(list(self._entries.values()))[:limit]

    def clear(self) -> None:
        self._entries.clear()


class FileLeaderboardRepository:
    def __init__(self, *, path: Path) -> None:
        self._path = path

    def record(self, entry: LeaderboardEntry) -> LeaderboardEntry:
        entries = {existing.run_id: existing for existing in self._read()}
        entries[entry.run_id] = entry
        self._write(_rank_entries(list(entries.values())))
        return entry

    def list_entries(self, *, limit: int = 20) -> list[LeaderboardEntry]:
        return _rank_entries(self._read())[:limit]

    def _read(self) -> list[LeaderboardEntry]:
        if not self._path.exists():
            return []
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        return [LeaderboardEntry(**item) for item in payload]

    def _write(self, entries: list[LeaderboardEntry]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps([asdict(entry) for entry in entries], indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _rank_entries(entries: list[LeaderboardEntry]) -> list[LeaderboardEntry]:
    return sorted(
        entries,
        key=lambda entry: (
            entry.ndcg,
            entry.mrr,
            entry.recall_at_k,
            entry.precision_at_k,
            -entry.avg_latency_ms,
        ),
        reverse=True,
    )
