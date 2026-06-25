from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class MetricSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    total_requests: int = Field(ge=0)
    successful_requests: int = Field(ge=0)
    failed_requests: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    error_rate: float = Field(ge=0, le=1)
    avg_retrieval_latency_ms: float = Field(ge=0)
    p95_retrieval_latency_ms: float = Field(ge=0)
    avg_chunks_returned: float = Field(ge=0)
    avg_similarity_score: float
    request_volume: int = Field(ge=0)
    trace_volume: int = Field(ge=0)
    failure_count: int = Field(ge=0)
