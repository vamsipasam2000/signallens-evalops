from collections import Counter

import pytest

from app.evaluators.dataset import load_eval_records
from app.evaluators.metrics import (
    accuracy,
    action_agreement,
    false_negative_rate,
    false_positive_rate,
    latency_summaries,
    macro_f1,
    summarize_latencies,
)
from app.evaluators.reports import build_eval_report_markdown, build_eval_summary, write_eval_report
from app.evaluators.runner import run_eval, summary_to_dict
from app.evaluators.types import EvalPrediction, EvalRecord


def _record(
    record_id: str,
    expected_risk: str,
    expected_action: str,
) -> EvalRecord:
    return EvalRecord(
        id=record_id,
        content=f"Example {record_id}",
        expected_risk=expected_risk,
        expected_action=expected_action,
        policy_area=expected_risk,
        severity="medium",
    )


def _prediction(
    record_id: str,
    risk_category: str,
    recommended_action: str,
    total_latency_ms: float = 8.0,
) -> EvalPrediction:
    return EvalPrediction(
        id=record_id,
        risk_category=risk_category,
        recommended_action=recommended_action,
        confidence=0.75,
        trace_id=f"trace-{record_id}",
        node_latencies_ms={"classify_risk": 1.0, "recommend_action": 0.5},
        total_latency_ms=total_latency_ms,
    )


def test_eval_dataset_is_small_labeled_and_balanced() -> None:
    records = load_eval_records()

    assert len(records) == 60
    assert 50 <= len(records) <= 75
    assert len({record.id for record in records}) == len(records)
    assert Counter(record.expected_risk for record in records) == {
        "safe": 15,
        "spam": 15,
        "harassment": 15,
        "self_harm_sensitive": 15,
    }


def test_metric_functions_compute_expected_values() -> None:
    records = [
        _record("r1", "safe", "allow"),
        _record("r2", "spam", "downrank"),
        _record("r3", "spam", "downrank"),
        _record("r4", "harassment", "block"),
    ]
    predictions = [
        _prediction("r1", "safe", "allow"),
        _prediction("r2", "spam", "downrank"),
        _prediction("r3", "safe", "allow"),
        _prediction("r4", "spam", "downrank"),
    ]

    assert accuracy(records, predictions) == 0.5
    assert action_agreement(records, predictions) == 0.5
    assert macro_f1(records, predictions) == 0.2917
    assert false_positive_rate(records, predictions) == 0.0
    assert false_negative_rate(records, predictions) == 0.3333


def test_latency_summaries_include_total_and_node_latency() -> None:
    predictions = [
        _prediction("r1", "safe", "allow", total_latency_ms=1.0),
        _prediction("r2", "spam", "downrank", total_latency_ms=2.0),
        _prediction("r3", "safe", "allow", total_latency_ms=10.0),
    ]

    total = summarize_latencies([1.0, 2.0, 10.0])
    summaries = latency_summaries(predictions)

    assert total.avg_ms == 4.3333
    assert total.p50_ms == 2.0
    assert total.p95_ms == 9.2
    assert summaries["total"].max_ms == 10.0
    assert summaries["classify_risk"].count == 3
    assert summaries["recommend_action"].avg_ms == 0.5


def test_metrics_reject_empty_or_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="At least one eval record"):
        accuracy([], [])

    with pytest.raises(ValueError, match="same length"):
        accuracy([_record("r1", "safe", "allow")], [])

    with pytest.raises(ValueError, match="At least one latency"):
        summarize_latencies([])


def test_eval_runner_scores_default_dataset_deterministically() -> None:
    summary = run_eval()

    assert summary.dataset_size == 60
    assert summary.accuracy == 0.8
    assert summary.macro_f1 == 0.8111
    assert summary.false_positive_rate == 0.2
    assert summary.false_negative_rate == 0.2
    assert summary.action_agreement == 0.8
    assert summary.confusion_matrix == {
        "safe": {"safe": 12, "spam": 3},
        "spam": {"spam": 12, "safe": 3},
        "harassment": {"harassment": 12, "safe": 3},
        "self_harm_sensitive": {"self_harm_sensitive": 12, "safe": 3},
    }
    assert set(summary.latency) == {
        "total",
        "normalize_content",
        "retrieve_policy_context",
        "classify_risk",
        "recommend_action",
        "generate_explanation",
        "evaluate_output",
    }
    assert summary.latency["total"].count == 60
    assert len(summary.predictions) == 60
    assert all(prediction.trace_id for prediction in summary.predictions)

    compact_summary = summary_to_dict(summary)
    assert "predictions" not in compact_summary
    assert compact_summary["dataset_size"] == 60


def test_eval_summary_and_report_generation(tmp_path) -> None:
    summary = run_eval()
    report_path = tmp_path / "week1_eval_summary.md"

    payload = build_eval_summary(summary, report_path=report_path)
    markdown = build_eval_report_markdown(summary)
    written_path = write_eval_report(summary, report_path=report_path)

    assert payload["metrics"]["accuracy"] == 0.8
    assert payload["report_path"].endswith("week1_eval_summary.md")
    assert "# SignalLens EvalOps Week 1 Evaluation Summary" in markdown
    assert "| Accuracy | `0.8000` |" in markdown
    assert "## Latency Summary" in markdown
    assert written_path == report_path
    assert report_path.read_text(encoding="utf-8") == markdown
