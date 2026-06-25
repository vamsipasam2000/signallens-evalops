from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.analysis.models import FailureType
from app.quality.models import QualityGateStatus, QualityMetricName


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


class RAGIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_name: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    content_type: str | None = Field(default="text/plain", max_length=200)
    source: str = Field(default="api", min_length=1, max_length=100)
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_size_tokens: int | None = Field(default=None, ge=1, le=4_000)
    chunk_overlap_tokens: int | None = Field(default=None, ge=0, le=2_000)

    @field_validator("document_name")
    @classmethod
    def document_name_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("document_name must not be blank")
        return normalized

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank_for_ingestion(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value


class RAGChunkResponse(BaseModel):
    chunk_id: str
    chunk_index: int
    token_count: int
    preview: str


class RAGIngestResponse(BaseModel):
    document_id: str
    name: str
    content_type: str
    status: Literal["ingested", "failed"]
    chunk_count: int
    token_count: int
    embedding_model: str
    parser_version: str
    chunker_version: str
    ingestion_latency_ms: float
    chunks: list[RAGChunkResponse]
    message: str


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=100)
    metadata_filter: dict[str, Any] = Field(default_factory=dict)
    strategy: Literal["cosine", "keyword_boosted"] = "cosine"

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    metadata: dict[str, Any]
    embedding_model: str
    score: float


class RetrievalResponse(BaseModel):
    trace_id: str | None
    query: str
    top_k: int
    strategy: Literal["cosine", "keyword_boosted"]
    backend: str
    embedding_model: str
    latency_ms: float
    results: list[RetrievedChunkResponse]
    status: Literal["completed"]
    message: str


class RetrievalGroundTruthItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(..., min_length=1, max_length=200)
    query: str = Field(..., min_length=1, max_length=2_000)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_document_ids: list[str] = Field(default_factory=list)
    metadata_filter: dict[str, Any] = Field(default_factory=dict)
    top_k: int | None = Field(default=None, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def evaluation_query_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


class RetrievalEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(default="api-ground-truth", min_length=1, max_length=200)
    run_name: str = Field(default="api-retrieval-evaluation", min_length=1, max_length=200)
    top_k: int = Field(default=5, ge=1, le=100)
    strategy: Literal["cosine", "keyword_boosted"] = "cosine"
    ground_truth: list[RetrievalGroundTruthItem] = Field(default_factory=list)
    run_benchmark_grid: bool = False


class RetrievalQueryMetricsResponse(BaseModel):
    query_id: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float
    latency_ms: float
    retrieved_count: int
    relevant_count: int
    avg_similarity_score: float = 0.0


class RetrievalEvaluationResponse(BaseModel):
    run_id: str
    dataset_size: int
    top_k: int
    strategy: Literal["cosine", "keyword_boosted"]
    backend: str
    embedding_model: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float
    avg_latency_ms: float
    per_query: list[RetrievalQueryMetricsResponse]
    report_path: str | None
    status: Literal["completed"]
    message: str


class RetrievalLeaderboardEntryResponse(BaseModel):
    run_id: str
    name: str
    dataset_name: str
    backend: str
    embedding_model: str
    retrieval_strategy: Literal["cosine", "keyword_boosted"]
    top_k: int
    chunk_size_tokens: int | None
    chunk_overlap_tokens: int | None
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float
    avg_latency_ms: float
    report_path: str | None
    parameters: dict[str, Any]


class RetrievalLeaderboardResponse(BaseModel):
    entries: list[RetrievalLeaderboardEntryResponse]
    status: Literal["completed"]
    message: str


class FailureAnalysisResponse(BaseModel):
    analysis_id: str
    timestamp: datetime
    query: str
    experiment_id: str
    failure_type: FailureType
    metric_name: str
    metric_value: float
    threshold: float
    recommendation: str


class QualityCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(..., min_length=1, max_length=200)
    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg: float = Field(ge=0, le=1)
    retrieval_latency_ms: float = Field(ge=0)
    similarity_score: float = Field(ge=0)


class QualityFailedCheckResponse(BaseModel):
    metric: QualityMetricName
    actual: float
    required: float
    reason: FailureType
    recommendation: str


class QualityGateResponse(BaseModel):
    gate_id: str
    timestamp: datetime
    experiment_id: str
    status: QualityGateStatus
    failed_checks: list[QualityFailedCheckResponse]
    metrics_snapshot: dict[str, float]


class DashboardSummaryResponse(BaseModel):
    total_traces: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    error_rate: float
    avg_retrieval_latency_ms: float
    p95_retrieval_latency_ms: float
    failed_quality_gates: int
    total_quality_checks: int


class DashboardLeaderboardEntryResponse(BaseModel):
    experiment_id: str
    name: str
    dataset_name: str
    embedding_model: str
    retriever: str
    strategy: str
    top_k: int
    chunk_size: int | None
    chunk_overlap: int | None
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float
    avg_retrieval_latency_ms: float
    quality_gate_status: QualityGateStatus | None


class DashboardExperimentResponse(BaseModel):
    experiment_id: str
    embedding_model: str
    chunk_size: int | None
    retriever: str
    top_k: int
    precision: float
    recall: float
    mrr: float
    ndcg: float
    quality_gate_status: QualityGateStatus | None
