from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.dependencies import get_trace_service
from app.core.errors import DependencyUnavailableError
from app.traces.models import (
    CreateTraceRequest,
    Trace,
    TraceListResponse,
    TraceSortField,
    TraceSortOrder,
    TraceStatus,
)
from app.traces.service import TraceListFilters, TraceService

router = APIRouter(tags=["traces"])


@router.post("/traces", response_model=Trace)
def create_trace(
    request: CreateTraceRequest,
    service: Annotated[TraceService, Depends(get_trace_service)],
) -> Trace:
    try:
        return service.record(request.to_trace())
    except DependencyUnavailableError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@router.get("/traces", response_model=TraceListResponse)
def list_traces(
    service: Annotated[TraceService, Depends(get_trace_service)],
    status: Annotated[TraceStatus | None, Query()] = None,
    retriever_name: Annotated[str | None, Query(min_length=1)] = None,
    embedding_model: Annotated[str | None, Query(min_length=1)] = None,
    start_date: Annotated[datetime | None, Query()] = None,
    end_date: Annotated[datetime | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    sort_by: Annotated[TraceSortField, Query()] = "timestamp",
    sort_order: Annotated[TraceSortOrder, Query()] = "desc",
) -> TraceListResponse:
    try:
        result = service.list(
            filters=TraceListFilters(
                status=status,
                retriever_name=retriever_name,
                embedding_model=embedding_model,
                start_date=start_date,
                end_date=end_date,
            ),
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order,
        )
    except DependencyUnavailableError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return TraceListResponse(
        items=result.items,
        total=result.total,
        limit=result.limit,
        offset=result.offset,
        sort_by=result.sort_by,
        sort_order=result.sort_order,
    )


@router.get("/traces/{trace_id}", response_model=Trace)
def get_trace(
    trace_id: str,
    service: Annotated[TraceService, Depends(get_trace_service)],
) -> Trace:
    try:
        trace = service.get(trace_id)
    except DependencyUnavailableError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace
