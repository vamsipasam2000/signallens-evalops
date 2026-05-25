from dataclasses import dataclass

from app.workflows.state import RecommendedAction, RiskCategory


@dataclass(frozen=True)
class EvalRecord:
    id: str
    content: str
    expected_risk: RiskCategory
    expected_action: RecommendedAction
    policy_area: RiskCategory
    severity: str


@dataclass(frozen=True)
class EvalPrediction:
    id: str
    risk_category: RiskCategory
    recommended_action: RecommendedAction
    confidence: float
    trace_id: str
    node_latencies_ms: dict[str, float]
    total_latency_ms: float


@dataclass(frozen=True)
class LatencySummary:
    count: int
    avg_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


@dataclass(frozen=True)
class EvalRunSummary:
    dataset_size: int
    accuracy: float
    macro_f1: float
    false_positive_rate: float
    false_negative_rate: float
    action_agreement: float
    latency: dict[str, LatencySummary]
    confusion_matrix: dict[str, dict[str, int]]
    predictions: list[EvalPrediction]

