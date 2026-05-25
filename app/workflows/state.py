from typing import Any, Literal, TypedDict

RiskCategory = Literal["safe", "spam", "harassment", "self_harm_sensitive"]
RecommendedAction = Literal["allow", "downrank", "human_review", "block"]


class WorkflowState(TypedDict, total=False):
    request_id: str
    trace_id: str
    tracing_enabled: bool
    raw_content: str
    normalized_content: str
    source: str | None
    metadata: dict[str, Any]
    input_length: int
    policy_context: str
    policy_categories: list[str]
    risk_category: RiskCategory
    matched_terms: list[str]
    confidence: float
    recommended_action: RecommendedAction
    explanation: str
    eval_scores: dict[str, float | bool | str]
    node_latencies_ms: dict[str, float]
