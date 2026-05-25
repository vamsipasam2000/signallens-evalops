from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter

from app.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    EvalSummaryResponse,
    HealthResponse,
    PolicyResponse,
)
from app.core.config import get_settings
from app.evaluators.reports import build_eval_summary, write_eval_report
from app.evaluators.runner import run_eval
from app.workflows.graph import run_workflow

router = APIRouter()

POLICY_PATH = Path(__file__).resolve().parents[1] / "data" / "policy.md"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        service=settings.app_name,
        status="ok",
        version=settings.app_version,
        environment=settings.app_env,
    )


@router.get("/v1/policy", response_model=PolicyResponse)
def get_policy() -> PolicyResponse:
    policy_text = POLICY_PATH.read_text(encoding="utf-8")
    return PolicyResponse(
        version="day1",
        categories=["safe", "spam", "harassment", "self_harm_sensitive"],
        policy=policy_text,
    )


@router.post("/v1/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    request_id = str(uuid4())
    result = run_workflow(
        {
            "request_id": request_id,
            "raw_content": request.content,
            "source": request.source,
            "metadata": request.metadata,
        }
    )

    return AnalyzeResponse(
        request_id=request_id,
        trace_id=result["trace_id"],
        tracing_enabled=result["tracing_enabled"],
        normalized_content=result["normalized_content"],
        risk_category=result["risk_category"],
        recommended_action=result["recommended_action"],
        confidence=result["confidence"],
        explanation=result["explanation"],
        eval_scores=result["eval_scores"],
        node_latencies_ms=result["node_latencies_ms"],
        workflow_version="day3-langgraph-langfuse-deterministic-v1",
        status="completed",
        message="Content analyzed with the Day 3 traced deterministic LangGraph workflow.",
    )


@router.post("/v1/evals/run", response_model=EvalSummaryResponse)
def run_evaluation() -> EvalSummaryResponse:
    summary = run_eval()
    report_path = write_eval_report(summary)
    payload = build_eval_summary(summary, report_path=report_path)

    return EvalSummaryResponse(
        **payload,
        status="completed",
        message="Evaluation completed and markdown summary written.",
    )


@router.get("/v1/evals/summary", response_model=EvalSummaryResponse)
def get_evaluation_summary() -> EvalSummaryResponse:
    summary = run_eval()
    payload = build_eval_summary(summary)

    return EvalSummaryResponse(
        **payload,
        status="completed",
        message="Evaluation summary generated from the deterministic local dataset.",
    )
