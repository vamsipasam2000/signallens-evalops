from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_metrics_service
from app.core.errors import DependencyUnavailableError
from app.metrics.models import MetricSnapshot
from app.metrics.service import MetricsService

router = APIRouter(tags=["metrics"])


@router.get("/metrics/summary", response_model=MetricSnapshot)
def get_metrics_summary(
    service: Annotated[MetricsService, Depends(get_metrics_service)],
) -> MetricSnapshot:
    try:
        return service.summarize()
    except DependencyUnavailableError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
