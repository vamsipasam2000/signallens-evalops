from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.schemas import QualityCheckRequest, QualityGateResponse
from app.core.dependencies import get_quality_gate_service
from app.quality.models import QualityMetricsSnapshot
from app.quality.service import QualityGateService

router = APIRouter(tags=["quality"])


@router.post("/quality/check", response_model=QualityGateResponse)
def check_quality(
    request: QualityCheckRequest,
    service: Annotated[QualityGateService, Depends(get_quality_gate_service)],
) -> QualityGateResponse:
    gate = service.evaluate_experiment(
        experiment_id=request.experiment_id,
        metrics_snapshot=QualityMetricsSnapshot(
            precision_at_k=request.precision_at_k,
            recall_at_k=request.recall_at_k,
            mrr=request.mrr,
            ndcg=request.ndcg,
            retrieval_latency_ms=request.retrieval_latency_ms,
            similarity_score=request.similarity_score,
        ),
    )
    return QualityGateResponse(**asdict(gate))


@router.get("/quality/checks", response_model=list[QualityGateResponse])
def list_quality_checks(
    service: Annotated[QualityGateService, Depends(get_quality_gate_service)],
    limit: int = Query(default=100, ge=1, le=500),
) -> list[QualityGateResponse]:
    return [
        QualityGateResponse(**asdict(gate))
        for gate in service.list_checks(limit=limit)
    ]
