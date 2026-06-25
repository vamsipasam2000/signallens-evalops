from __future__ import annotations

import json
from typing import Any, Protocol

from app.core.errors import DependencyUnavailableError
from app.traces.models import Trace


class TraceRepository(Protocol):
    def save(self, trace: Trace) -> Trace:
        ...

    def get(self, trace_id: str) -> Trace | None:
        ...

    def list_all(self) -> list[Trace]:
        ...


class InMemoryTraceRepository:
    def __init__(self) -> None:
        self._traces: dict[str, Trace] = {}

    def save(self, trace: Trace) -> Trace:
        stored = trace.model_copy(deep=True)
        self._traces[stored.trace_id] = stored
        return stored.model_copy(deep=True)

    def get(self, trace_id: str) -> Trace | None:
        trace = self._traces.get(trace_id)
        return trace.model_copy(deep=True) if trace is not None else None

    def list_all(self) -> list[Trace]:
        return [
            trace.model_copy(deep=True)
            for trace in sorted(self._traces.values(), key=lambda item: item.timestamp)
        ]

    def clear(self) -> None:
        self._traces.clear()


class PostgresTraceRepository:
    """PostgreSQL-ready JSONB trace repository.

    This adapter keeps the interface production-shaped without requiring
    PostgreSQL for local tests.
    """

    def __init__(self, *, dsn: str, table_name: str = "signallens_traces") -> None:
        try:
            import psycopg
        except ModuleNotFoundError as exc:
            raise DependencyUnavailableError(
                "psycopg is not installed. Install with `pip install -e '.[platform]'`."
            ) from exc

        self._psycopg = psycopg
        self._dsn = dsn
        self._table_name = _safe_identifier(table_name)
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with self._psycopg.connect(self._dsn) as connection:
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    trace_id text PRIMARY KEY,
                    timestamp timestamptz NOT NULL,
                    query text NOT NULL,
                    retriever_name text NOT NULL,
                    embedding_model text NOT NULL,
                    status text NOT NULL,
                    payload jsonb NOT NULL
                )
                """
            )
            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._table_name}_timestamp_idx
                ON {self._table_name} (timestamp DESC)
                """
            )
            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._table_name}_status_idx
                ON {self._table_name} (status)
                """
            )

    def save(self, trace: Trace) -> Trace:
        payload = trace.model_dump(mode="json")
        with self._psycopg.connect(self._dsn) as connection:
            connection.execute(
                f"""
                INSERT INTO {self._table_name}
                    (trace_id, timestamp, query, retriever_name, embedding_model, status, payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (trace_id) DO UPDATE SET
                    timestamp = EXCLUDED.timestamp,
                    query = EXCLUDED.query,
                    retriever_name = EXCLUDED.retriever_name,
                    embedding_model = EXCLUDED.embedding_model,
                    status = EXCLUDED.status,
                    payload = EXCLUDED.payload
                """,
                (
                    trace.trace_id,
                    trace.timestamp,
                    trace.query,
                    trace.retriever_name,
                    trace.embedding_model,
                    trace.status,
                    json.dumps(payload, sort_keys=True),
                ),
            )
        return trace.model_copy(deep=True)

    def get(self, trace_id: str) -> Trace | None:
        with self._psycopg.connect(self._dsn) as connection:
            row = connection.execute(
                f"SELECT payload FROM {self._table_name} WHERE trace_id = %s",
                (trace_id,),
            ).fetchone()
        if row is None:
            return None
        return _trace_from_payload(row[0])

    def list_all(self) -> list[Trace]:
        with self._psycopg.connect(self._dsn) as connection:
            rows = connection.execute(
                f"SELECT payload FROM {self._table_name} ORDER BY timestamp ASC",
            ).fetchall()
        return [_trace_from_payload(row[0]) for row in rows]


def _safe_identifier(value: str) -> str:
    if not value or not value.replace("_", "").isalnum() or value[0].isdigit():
        raise ValueError("table_name must be a safe SQL identifier")
    return value


def _trace_from_payload(payload: Any) -> Trace:
    if isinstance(payload, str):
        return Trace(**json.loads(payload))
    return Trace(**dict(payload))
