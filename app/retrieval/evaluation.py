from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.core.errors import RetrievalValidationError
from app.ranking.metrics import (
    mean_reciprocal_rank_from_flags,
    ndcg_at_k_from_flags,
    precision_at_k_from_flags,
    recall_at_k_from_flags,
)
from app.retrieval.models import (
    GroundTruthQuery,
    LeaderboardEntry,
    RetrievalEvaluationSummary,
    RetrievalQuery,
    RetrievalQueryMetrics,
    RetrievalStrategy,
    RetrievedChunk,
)
from app.retrieval.service import RetrievalService


class RetrievalEvaluationService:
    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        backend: str,
        embedding_model: str,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._backend = backend
        self._embedding_model = embedding_model

    def evaluate(
        self,
        *,
        records: list[GroundTruthQuery],
        top_k: int,
        strategy: RetrievalStrategy,
        run_id: str | None = None,
        trace_request_payload: dict[str, Any] | None = None,
    ) -> RetrievalEvaluationSummary:
        if not records:
            raise RetrievalValidationError("At least one ground-truth query is required.")
        if top_k <= 0:
            raise RetrievalValidationError("top_k must be greater than zero")

        query_metrics: list[RetrievalQueryMetrics] = []
        for record in records:
            if not record.relevant_chunk_ids and not record.relevant_document_ids:
                raise RetrievalValidationError(
                    f"Ground-truth query {record.query_id} has no relevant ids."
                )

            effective_top_k = record.top_k or top_k
            result = self._retrieval_service.retrieve(
                RetrievalQuery(
                    query=record.query,
                    top_k=effective_top_k,
                    metadata_filter=record.metadata_filter,
                    strategy=strategy,
                ),
                trace_request_payload=_per_query_trace_payload(
                    base_payload=trace_request_payload,
                    record=record,
                    effective_top_k=effective_top_k,
                    strategy=strategy,
                ),
            )
            relevance_flags = _relevance_flags(result.results, record)
            total_relevant = _total_relevant(record)
            similarity_scores = [chunk.score for chunk in result.results]
            query_metrics.append(
                RetrievalQueryMetrics(
                    query_id=record.query_id,
                    precision_at_k=precision_at_k_from_flags(
                        relevance_flags,
                        effective_top_k,
                    ),
                    recall_at_k=recall_at_k_from_flags(
                        relevance_flags,
                        total_relevant,
                        effective_top_k,
                    ),
                    mrr=mean_reciprocal_rank_from_flags(relevance_flags),
                    ndcg=ndcg_at_k_from_flags(
                        relevance_flags,
                        total_relevant=total_relevant,
                        k=effective_top_k,
                    ),
                    latency_ms=result.latency_ms,
                    retrieved_count=len(result.results),
                    relevant_count=total_relevant,
                    avg_similarity_score=_mean(similarity_scores),
                )
            )

        return RetrievalEvaluationSummary(
            run_id=run_id or str(uuid4()),
            dataset_size=len(records),
            top_k=top_k,
            strategy=strategy,
            backend=self._backend,
            embedding_model=self._embedding_model,
            precision_at_k=_mean([metric.precision_at_k for metric in query_metrics]),
            recall_at_k=_mean([metric.recall_at_k for metric in query_metrics]),
            mrr=_mean([metric.mrr for metric in query_metrics]),
            ndcg=_mean([metric.ndcg for metric in query_metrics]),
            avg_latency_ms=_mean([metric.latency_ms for metric in query_metrics], digits=3),
            per_query=query_metrics,
            avg_similarity_score=_mean(
                [metric.avg_similarity_score for metric in query_metrics]
            ),
        )

    def to_leaderboard_entry(
        self,
        *,
        summary: RetrievalEvaluationSummary,
        name: str,
        dataset_name: str,
        chunk_size_tokens: int | None = None,
        chunk_overlap_tokens: int | None = None,
        report_path: str | None = None,
        parameters: dict[str, object] | None = None,
    ) -> LeaderboardEntry:
        return LeaderboardEntry(
            run_id=summary.run_id,
            name=name,
            dataset_name=dataset_name,
            backend=summary.backend,
            embedding_model=summary.embedding_model,
            retrieval_strategy=summary.strategy,
            top_k=summary.top_k,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            precision_at_k=summary.precision_at_k,
            recall_at_k=summary.recall_at_k,
            mrr=summary.mrr,
            ndcg=summary.ndcg,
            avg_latency_ms=summary.avg_latency_ms,
            report_path=report_path,
            parameters=dict(parameters or {}),
        )


def _mean(values: list[float], *, digits: int = 4) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), digits)


def _relevance_flags(
    chunks: list[RetrievedChunk],
    record: GroundTruthQuery,
) -> list[bool]:
    seen_chunk_ids: set[str] = set()
    seen_document_ids: set[str] = set()
    flags: list[bool] = []

    for chunk in chunks:
        is_relevant_chunk = chunk.chunk_id in record.relevant_chunk_ids
        is_relevant_document = chunk.document_id in record.relevant_document_ids

        if is_relevant_chunk and chunk.chunk_id not in seen_chunk_ids:
            flags.append(True)
            seen_chunk_ids.add(chunk.chunk_id)
            continue

        if is_relevant_document and chunk.document_id not in seen_document_ids:
            flags.append(True)
            seen_document_ids.add(chunk.document_id)
            continue

        flags.append(False)

    return flags


def _total_relevant(record: GroundTruthQuery) -> int:
    return len(record.relevant_chunk_ids) + len(record.relevant_document_ids)


def _per_query_trace_payload(
    *,
    base_payload: dict[str, Any] | None,
    record: GroundTruthQuery,
    effective_top_k: int,
    strategy: RetrievalStrategy,
) -> dict[str, Any]:
    payload = dict(base_payload or {})
    payload["retrieval_query"] = {
        "query_id": record.query_id,
        "query": record.query,
        "top_k": effective_top_k,
        "metadata_filter": dict(record.metadata_filter),
        "strategy": strategy,
        "relevant_chunk_ids": sorted(record.relevant_chunk_ids),
        "relevant_document_ids": sorted(record.relevant_document_ids),
    }
    return payload
