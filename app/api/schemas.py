from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    service: str
    status: Literal["ok"]
    version: str
    environment: str


class PolicyResponse(BaseModel):
    version: str
    categories: list[str]
    policy: str


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1, max_length=5_000)
    source: str | None = Field(default=None, max_length=100)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("content must not be blank")
        return normalized


class AnalyzeResponse(BaseModel):
    request_id: str
    trace_id: str
    tracing_enabled: bool
    normalized_content: str
    risk_category: Literal["safe", "spam", "harassment", "self_harm_sensitive"]
    recommended_action: Literal["allow", "downrank", "human_review", "block"]
    confidence: float
    explanation: str
    eval_scores: dict[str, Any]
    node_latencies_ms: dict[str, float]
    workflow_version: str
    status: Literal["completed"]
    message: str


class EvalMetricsResponse(BaseModel):
    accuracy: float
    macro_f1: float
    false_positive_rate: float
    false_negative_rate: float
    action_agreement: float


class LatencySummaryResponse(BaseModel):
    count: int
    avg_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


class EvalSummaryResponse(BaseModel):
    status: Literal["completed"]
    dataset_size: int
    dataset_path: str
    workflow_version: str
    metrics: EvalMetricsResponse
    latency: dict[str, LatencySummaryResponse]
    confusion_matrix: dict[str, dict[str, int]]
    report_path: str
    message: str
