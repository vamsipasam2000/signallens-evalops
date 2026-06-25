from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.analysis.service import ErrorAnalysisService
from app.api.schemas import (
    RetrievalEvaluationRequest,
    RetrievalEvaluationResponse,
    RetrievalGroundTruthItem,
    RetrievalLeaderboardEntryResponse,
    RetrievalLeaderboardResponse,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunkResponse,
)
from app.core.dependencies import (
    get_error_analysis_service,
    get_leaderboard_repository,
    get_quality_gate_service,
    get_retrieval_benchmark_suite,
    get_retrieval_evaluation_service,
    get_retrieval_service,
    get_trace_repository,
)
from app.core.errors import DependencyUnavailableError, RetrievalValidationError
from app.quality.service import QualityGateService
from app.retrieval.benchmark import RetrievalBenchmarkSuite, load_ground_truth_records
from app.retrieval.evaluation import RetrievalEvaluationService
from app.retrieval.leaderboard import LeaderboardRepository
from app.retrieval.models import GroundTruthQuery, RetrievalEvaluationSummary, RetrievalQuery
from app.retrieval.service import RetrievalService
from app.traces.models import Trace
from app.traces.repositories import TraceRepository

router = APIRouter(tags=["retrieval"])


@router.post("/retrieve", response_model=RetrievalResponse)
def retrieve(
    request: RetrievalRequest,
    service: Annotated[RetrievalService, Depends(get_retrieval_service)],
) -> RetrievalResponse:
    request_payload = _endpoint_request_payload("/retrieve", request.model_dump(mode="json"))
    try:
        result = service.retrieve(
            RetrievalQuery(
                query=request.query,
                top_k=request.top_k,
                metadata_filter=request.metadata_filter,
                strategy=request.strategy,
            ),
            trace_request_payload=request_payload,
        )
    except RetrievalValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DependencyUnavailableError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    return RetrievalResponse(
        trace_id=result.trace_id,
        query=result.query,
        top_k=result.top_k,
        strategy=result.strategy,
        backend=result.backend,
        embedding_model=result.embedding_model,
        latency_ms=result.latency_ms,
        results=[
            RetrievedChunkResponse(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                token_count=chunk.token_count,
                metadata=chunk.metadata,
                embedding_model=chunk.embedding_model,
                score=chunk.score,
            )
            for chunk in result.results
        ],
        status="completed",
        message="Retrieved ranked chunks with latency and similarity scores.",
    )


@router.post("/evaluate/retrieval", response_model=RetrievalEvaluationResponse)
def evaluate_retrieval(
    request: RetrievalEvaluationRequest,
    evaluator: Annotated[
        RetrievalEvaluationService,
        Depends(get_retrieval_evaluation_service),
    ],
    leaderboard_repository: Annotated[
        LeaderboardRepository,
        Depends(get_leaderboard_repository),
    ],
    benchmark_suite: Annotated[
        RetrievalBenchmarkSuite,
        Depends(get_retrieval_benchmark_suite),
    ],
    trace_repository: Annotated[TraceRepository, Depends(get_trace_repository)],
    analysis_service: Annotated[
        ErrorAnalysisService,
        Depends(get_error_analysis_service),
    ],
    quality_service: Annotated[
        QualityGateService,
        Depends(get_quality_gate_service),
    ],
) -> RetrievalEvaluationResponse:
    started_at = perf_counter()
    request_payload = _endpoint_request_payload(
        "/evaluate/retrieval",
        request.model_dump(mode="json"),
    )
    try:
        if request.run_benchmark_grid or not request.ground_truth:
            analysis_records = load_ground_truth_records()
            results = benchmark_suite.run(
                dataset_name=request.dataset_name,
                ground_truth=analysis_records,
                trace_request_payload=request_payload,
            )
            for result in results:
                analysis_service.analyze_retrieval_evaluation(
                    summary=result.evaluation,
                    records=analysis_records,
                )
                quality_service.evaluate_retrieval_summary(result.evaluation)
            best = sorted(
                results,
                key=lambda result: (
                    result.evaluation.ndcg,
                    result.evaluation.mrr,
                    result.evaluation.recall_at_k,
                    result.evaluation.precision_at_k,
                ),
                reverse=True,
            )[0]
            return _evaluation_response(
                best.evaluation,
                message="Retrieval benchmark grid completed and leaderboard updated.",
            )

        ground_truth_records = _ground_truth_records(request.ground_truth)
        summary = evaluator.evaluate(
            records=ground_truth_records,
            top_k=request.top_k,
            strategy=request.strategy,
            trace_request_payload=request_payload,
        )
        analysis_service.analyze_retrieval_evaluation(
            summary=summary,
            records=ground_truth_records,
        )
        quality_service.evaluate_retrieval_summary(summary)
        entry = evaluator.to_leaderboard_entry(
            summary=summary,
            name=request.run_name,
            dataset_name=request.dataset_name,
            parameters={"source": "api"},
        )
        leaderboard_repository.record(entry)
    except RetrievalValidationError as exc:
        _record_failed_evaluation_trace(
            trace_repository=trace_repository,
            request=request,
            request_payload=request_payload,
            error=str(exc),
            started_at=started_at,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DependencyUnavailableError as exc:
        _record_failed_evaluation_trace(
            trace_repository=trace_repository,
            request=request,
            request_payload=request_payload,
            error=str(exc),
            started_at=started_at,
        )
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    return _evaluation_response(
        summary,
        message="Retrieval evaluation completed and leaderboard updated.",
    )


@router.get("/leaderboard", response_model=RetrievalLeaderboardResponse)
def leaderboard(
    leaderboard_repository: Annotated[
        LeaderboardRepository,
        Depends(get_leaderboard_repository),
    ],
    limit: int = Query(default=20, ge=1, le=100),
) -> RetrievalLeaderboardResponse:
    entries = leaderboard_repository.list_entries(limit=limit)
    return RetrievalLeaderboardResponse(
        entries=[
            RetrievalLeaderboardEntryResponse(**asdict(entry))
            for entry in entries
        ],
        status="completed",
        message="Retrieval leaderboard returned in ranking order.",
    )


def _ground_truth_records(items: list[RetrievalGroundTruthItem]) -> list[GroundTruthQuery]:
    return [
        GroundTruthQuery(
            query_id=item.query_id,
            query=item.query,
            relevant_chunk_ids=set(item.relevant_chunk_ids),
            relevant_document_ids=set(item.relevant_document_ids),
            metadata_filter=item.metadata_filter,
            top_k=item.top_k,
        )
        for item in items
    ]


def _evaluation_response(
    summary: RetrievalEvaluationSummary,
    *,
    message: str,
) -> RetrievalEvaluationResponse:
    return RetrievalEvaluationResponse(
        run_id=summary.run_id,
        dataset_size=summary.dataset_size,
        top_k=summary.top_k,
        strategy=summary.strategy,
        backend=summary.backend,
        embedding_model=summary.embedding_model,
        precision_at_k=summary.precision_at_k,
        recall_at_k=summary.recall_at_k,
        mrr=summary.mrr,
        ndcg=summary.ndcg,
        avg_latency_ms=summary.avg_latency_ms,
        per_query=[
            {
                "query_id": metric.query_id,
                "precision_at_k": metric.precision_at_k,
                "recall_at_k": metric.recall_at_k,
                "mrr": metric.mrr,
                "ndcg": metric.ndcg,
                "latency_ms": metric.latency_ms,
                "retrieved_count": metric.retrieved_count,
                "relevant_count": metric.relevant_count,
                "avg_similarity_score": metric.avg_similarity_score,
            }
            for metric in summary.per_query
        ],
        report_path=summary.report_path,
        status="completed",
        message=message,
    )


def _endpoint_request_payload(endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "endpoint": endpoint,
        "body": body,
    }


def _record_failed_evaluation_trace(
    *,
    trace_repository: TraceRepository,
    request: RetrievalEvaluationRequest,
    request_payload: dict[str, Any],
    error: str,
    started_at: float,
) -> None:
    latency_ms = round((perf_counter() - started_at) * 1000, 3)
    trace_repository.save(
        Trace(
            request_payload=request_payload,
            query=_evaluation_request_query(request),
            retriever_name="retrieval-evaluation",
            embedding_model="unknown",
            retrieved_chunks=[],
            similarity_scores=[],
            retrieval_latency_ms=latency_ms,
            generation_latency_ms=0.0,
            total_latency_ms=latency_ms,
            status="failed",
            error_message=error,
        )
    )


def _evaluation_request_query(request: RetrievalEvaluationRequest) -> str:
    if request.ground_truth:
        return request.ground_truth[0].query
    return request.dataset_name
