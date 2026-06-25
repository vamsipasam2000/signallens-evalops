from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any

from app.core.errors import RetrievalValidationError
from app.retrieval.models import RetrievalQuery, RetrievalResult
from app.retrieval.retrievers import Retriever
from app.traces.models import Trace, TraceChunk
from app.traces.repositories import TraceRepository


class RetrievalService:
    def __init__(
        self,
        *,
        retriever: Retriever,
        trace_repository: TraceRepository | None = None,
        embedding_model: str = "unknown",
    ) -> None:
        self._retriever = retriever
        self._trace_repository = trace_repository
        self._embedding_model = embedding_model

    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        trace_request_payload: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        started_at = perf_counter()
        request_payload = trace_request_payload or _retrieval_query_payload(query)
        if not query.query.strip():
            error = "query must not be blank"
            self._record_failed_trace(
                query=query.query,
                request_payload=request_payload,
                error=error,
                started_at=started_at,
            )
            raise RetrievalValidationError(error)
        if query.top_k <= 0:
            error = "top_k must be greater than zero"
            self._record_failed_trace(
                query=query.query,
                request_payload=request_payload,
                error=error,
                started_at=started_at,
            )
            raise RetrievalValidationError(error)
        try:
            result = self._retriever.retrieve(query)
        except ValueError as exc:
            self._record_failed_trace(
                query=query.query,
                request_payload=request_payload,
                error=str(exc),
                started_at=started_at,
            )
            raise RetrievalValidationError(str(exc)) from exc
        except Exception as exc:
            self._record_failed_trace(
                query=query.query,
                request_payload=request_payload,
                error=str(exc),
                started_at=started_at,
            )
            raise
        trace = self._record_completed_trace(result, request_payload=request_payload)
        if trace is None:
            return result
        return replace(result, trace_id=trace.trace_id)

    def _record_completed_trace(
        self,
        result: RetrievalResult,
        *,
        request_payload: dict[str, Any],
    ) -> Trace | None:
        if self._trace_repository is None:
            return None

        trace = Trace(
            request_payload=request_payload,
            query=result.query,
            retriever_name=result.backend,
            embedding_model=result.embedding_model,
            retrieved_chunks=[
                TraceChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=chunk.score,
                )
                for chunk in result.results
            ],
            similarity_scores=[chunk.score for chunk in result.results],
            retrieval_latency_ms=result.latency_ms,
            generation_latency_ms=0.0,
            total_latency_ms=result.latency_ms,
            status="completed",
            error_message=None,
        )
        return self._trace_repository.save(trace)

    def _record_failed_trace(
        self,
        *,
        query: str,
        request_payload: dict[str, Any],
        error: str,
        started_at: float,
    ) -> None:
        if self._trace_repository is None:
            return
        latency_ms = round((perf_counter() - started_at) * 1000, 3)
        trace = Trace(
            request_payload=request_payload,
            query=query or "<blank>",
            retriever_name=self._retriever.backend_name,
            embedding_model=self._embedding_model,
            retrieved_chunks=[],
            similarity_scores=[],
            retrieval_latency_ms=latency_ms,
            generation_latency_ms=0.0,
            total_latency_ms=latency_ms,
            status="failed",
            error_message=error,
        )
        self._trace_repository.save(trace)


def _retrieval_query_payload(query: RetrievalQuery) -> dict[str, Any]:
    return {
        "query": query.query,
        "top_k": query.top_k,
        "metadata_filter": dict(query.metadata_filter),
        "strategy": query.strategy,
    }
