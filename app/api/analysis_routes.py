from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.analysis.service import ErrorAnalysisService
from app.api.schemas import FailureAnalysisResponse
from app.core.dependencies import get_error_analysis_service

router = APIRouter(tags=["analysis"])


@router.get("/analysis/failures", response_model=list[FailureAnalysisResponse])
def list_failure_analysis(
    service: Annotated[ErrorAnalysisService, Depends(get_error_analysis_service)],
    limit: int = Query(default=100, ge=1, le=500),
) -> list[FailureAnalysisResponse]:
    return [
        FailureAnalysisResponse(**asdict(analysis))
        for analysis in service.list_failures(limit=limit)
    ]
