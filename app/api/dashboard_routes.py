from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.schemas import (
    DashboardExperimentResponse,
    DashboardLeaderboardEntryResponse,
    DashboardSummaryResponse,
)
from app.core.dependencies import get_dashboard_service
from app.dashboard.models import DashboardFilters, DashboardSortField, DashboardSortOrder
from app.dashboard.service import DashboardService
from app.quality.models import QualityGateStatus

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard/summary", response_model=DashboardSummaryResponse)
def dashboard_summary(
    service: Annotated[DashboardService, Depends(get_dashboard_service)],
) -> DashboardSummaryResponse:
    return DashboardSummaryResponse(**asdict(service.summary()))


@router.get(
    "/dashboard/leaderboard",
    response_model=list[DashboardLeaderboardEntryResponse],
)
def dashboard_leaderboard(
    service: Annotated[DashboardService, Depends(get_dashboard_service)],
    limit: int = Query(default=20, ge=1, le=100),
    sort_by: Annotated[DashboardSortField, Query()] = "ndcg",
    sort_order: Annotated[DashboardSortOrder, Query()] = "desc",
    experiment_id: Annotated[str | None, Query()] = None,
    embedding_model: Annotated[str | None, Query()] = None,
    retriever: Annotated[str | None, Query()] = None,
    quality_gate_status: Annotated[QualityGateStatus | None, Query()] = None,
) -> list[DashboardLeaderboardEntryResponse]:
    entries = service.leaderboard(
        limit=limit,
        sort_by=sort_by,
        sort_order=sort_order,
        filters=DashboardFilters(
            experiment_id=experiment_id,
            embedding_model=embedding_model,
            retriever=retriever,
            quality_gate_status=quality_gate_status,
        ),
    )
    return [DashboardLeaderboardEntryResponse(**asdict(entry)) for entry in entries]


@router.get(
    "/dashboard/experiments",
    response_model=list[DashboardExperimentResponse],
)
def dashboard_experiments(
    service: Annotated[DashboardService, Depends(get_dashboard_service)],
    limit: int = Query(default=100, ge=1, le=500),
    experiment_id: Annotated[str | None, Query()] = None,
    embedding_model: Annotated[str | None, Query()] = None,
    retriever: Annotated[str | None, Query()] = None,
    quality_gate_status: Annotated[QualityGateStatus | None, Query()] = None,
) -> list[DashboardExperimentResponse]:
    experiments = service.experiments(
        limit=limit,
        filters=DashboardFilters(
            experiment_id=experiment_id,
            embedding_model=embedding_model,
            retriever=retriever,
            quality_gate_status=quality_gate_status,
        ),
    )
    return [
        DashboardExperimentResponse(**asdict(experiment))
        for experiment in experiments
    ]
