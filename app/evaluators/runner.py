import json
from dataclasses import asdict
from time import perf_counter

from app.evaluators.dataset import load_eval_records
from app.evaluators.metrics import (
    accuracy,
    action_agreement,
    confusion_counts,
    false_negative_rate,
    false_positive_rate,
    latency_summaries,
    macro_f1,
)
from app.evaluators.types import EvalPrediction, EvalRecord, EvalRunSummary
from app.workflows.graph import run_workflow


def run_eval(records: list[EvalRecord] | None = None) -> EvalRunSummary:
    eval_records = records if records is not None else load_eval_records()
    if not eval_records:
        raise ValueError("At least one eval record is required.")

    predictions: list[EvalPrediction] = []

    for record in eval_records:
        started_at = perf_counter()
        result = run_workflow(
            {
                "raw_content": record.content,
                "source": "eval-runner",
                "metadata": {
                    "eval_id": record.id,
                    "expected_risk": record.expected_risk,
                    "expected_action": record.expected_action,
                    "policy_area": record.policy_area,
                    "severity": record.severity,
                },
            }
        )
        total_latency_ms = round((perf_counter() - started_at) * 1000, 3)
        predictions.append(
            EvalPrediction(
                id=record.id,
                risk_category=result["risk_category"],
                recommended_action=result["recommended_action"],
                confidence=result["confidence"],
                trace_id=result["trace_id"],
                node_latencies_ms=result["node_latencies_ms"],
                total_latency_ms=total_latency_ms,
            )
        )

    return EvalRunSummary(
        dataset_size=len(eval_records),
        accuracy=accuracy(eval_records, predictions),
        macro_f1=macro_f1(eval_records, predictions),
        false_positive_rate=false_positive_rate(eval_records, predictions),
        false_negative_rate=false_negative_rate(eval_records, predictions),
        action_agreement=action_agreement(eval_records, predictions),
        latency=latency_summaries(predictions),
        confusion_matrix=confusion_counts(eval_records, predictions),
        predictions=predictions,
    )


def summary_to_dict(summary: EvalRunSummary) -> dict[str, object]:
    return {
        "dataset_size": summary.dataset_size,
        "accuracy": summary.accuracy,
        "macro_f1": summary.macro_f1,
        "false_positive_rate": summary.false_positive_rate,
        "false_negative_rate": summary.false_negative_rate,
        "action_agreement": summary.action_agreement,
        "latency": {
            name: asdict(latency_summary)
            for name, latency_summary in summary.latency.items()
        },
        "confusion_matrix": summary.confusion_matrix,
    }


def main() -> None:
    print(json.dumps(summary_to_dict(run_eval()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
