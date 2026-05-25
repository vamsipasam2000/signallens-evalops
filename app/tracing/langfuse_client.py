from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Any
from uuid import uuid4

from app.core.config import get_settings

WORKFLOW_VERSION = "day3-langgraph-langfuse-deterministic-v1"


@dataclass
class ActiveTrace:
    trace_id: str
    enabled: bool
    client: Any | None = None
    root_span: Any | None = None

    def update_output(self, output: dict[str, Any], metadata: dict[str, Any]) -> None:
        if not self.enabled or self.root_span is None:
            return
        self.root_span.update(output=output, metadata=metadata)

    def log_score(self, name: str, value: float | str, comment: str | None = None) -> None:
        if not self.enabled or self.client is None:
            return
        data_type = "NUMERIC" if isinstance(value, int | float) else "CATEGORICAL"
        self.client.score_current_trace(
            name=name,
            value=value,
            data_type=data_type,
            comment=comment,
            metadata={"workflow_version": WORKFLOW_VERSION},
        )

    def flush(self) -> None:
        if self.enabled and self.client is not None:
            self.client.flush()


@lru_cache
def _get_langfuse_client() -> Any | None:
    settings = get_settings()
    if not settings.langfuse_enabled:
        return None

    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        environment=settings.app_env,
        release=settings.app_version,
    )


@contextmanager
def request_trace(input_payload: dict[str, Any], metadata: dict[str, Any]) -> Iterator[ActiveTrace]:
    client = _get_langfuse_client()
    local_trace_id = uuid4().hex

    if client is None:
        yield ActiveTrace(trace_id=local_trace_id, enabled=False)
        return

    with client.start_as_current_observation(
        as_type="span",
        name="signalens.analyze",
        input=input_payload,
        metadata=metadata,
        version=WORKFLOW_VERSION,
    ) as root_span:
        trace_id = client.get_current_trace_id() or local_trace_id
        active_trace = ActiveTrace(
            trace_id=trace_id,
            enabled=True,
            client=client,
            root_span=root_span,
        )
        try:
            yield active_trace
        finally:
            active_trace.flush()


@contextmanager
def node_span(
    name: str,
    state: dict[str, Any],
    input_payload: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    client = _get_langfuse_client()
    span_metadata = {
        "request_id": state.get("request_id"),
        "trace_id": state.get("trace_id"),
        "workflow_version": WORKFLOW_VERSION,
        "node": name,
    }

    timing = {"started_at": perf_counter(), "latency_ms": 0.0}

    context = (
        client.start_as_current_observation(
            as_type="span",
            name=name,
            input=input_payload,
            metadata=span_metadata,
            version=WORKFLOW_VERSION,
        )
        if client is not None
        else nullcontext(None)
    )

    with context as span:
        timing["span"] = span
        yield timing


def finish_node_span(
    timing: dict[str, Any],
    output_payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> float:
    latency_ms = round((perf_counter() - timing["started_at"]) * 1000, 3)
    timing["latency_ms"] = latency_ms

    span = timing.get("span")
    if span is not None:
        span.update(
            output=output_payload,
            metadata={
                "latency_ms": latency_ms,
                **(metadata or {}),
            },
        )

    return latency_ms


def score_eval_outputs(active_trace: ActiveTrace, eval_scores: dict[str, Any]) -> None:
    for name, value in eval_scores.items():
        if isinstance(value, bool):
            active_trace.log_score(name=name, value=1.0 if value else 0.0)
        elif isinstance(value, int | float):
            active_trace.log_score(name=name, value=float(value))
        elif isinstance(value, str):
            active_trace.log_score(name=name, value=value)
