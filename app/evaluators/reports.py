from dataclasses import asdict
from pathlib import Path

from app.evaluators.dataset import DEFAULT_EVAL_SET_PATH
from app.evaluators.types import EvalRunSummary
from app.tracing.langfuse_client import WORKFLOW_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = PROJECT_ROOT / "reports" / "week1_eval_summary.md"


def build_eval_summary(
    summary: EvalRunSummary,
    report_path: Path = REPORT_PATH,
) -> dict[str, object]:
    return {
        "dataset_size": summary.dataset_size,
        "dataset_path": _relative_path(DEFAULT_EVAL_SET_PATH),
        "workflow_version": WORKFLOW_VERSION,
        "metrics": {
            "accuracy": summary.accuracy,
            "macro_f1": summary.macro_f1,
            "false_positive_rate": summary.false_positive_rate,
            "false_negative_rate": summary.false_negative_rate,
            "action_agreement": summary.action_agreement,
        },
        "latency": {
            name: asdict(latency_summary)
            for name, latency_summary in summary.latency.items()
        },
        "confusion_matrix": summary.confusion_matrix,
        "report_path": _relative_path(report_path),
    }


def build_eval_report_markdown(summary: EvalRunSummary) -> str:
    eval_summary = build_eval_summary(summary)
    metrics = eval_summary["metrics"]
    latency = eval_summary["latency"]
    confusion_matrix = eval_summary["confusion_matrix"]

    return "\n".join(
        [
            "# SignalLens EvalOps Week 1 Evaluation Summary",
            "",
            "## Scope",
            "",
            "- Dataset: `app/data/eval_set.jsonl`",
            f"- Eval records: `{summary.dataset_size}`",
            f"- Workflow version: `{WORKFLOW_VERSION}`",
            "- Runner: deterministic local LangGraph workflow",
            "- Model calls: none",
            "- Databases: none",
            "- Docker: not included in Week 1",
            "- AWS: not included in Week 1",
            "",
            "## Aggregate Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Accuracy | `{metrics['accuracy']:.4f}` |",
            f"| Macro F1 | `{metrics['macro_f1']:.4f}` |",
            f"| False positive rate | `{metrics['false_positive_rate']:.4f}` |",
            f"| False negative rate | `{metrics['false_negative_rate']:.4f}` |",
            f"| Action agreement | `{metrics['action_agreement']:.4f}` |",
            "",
            "## Confusion Matrix",
            "",
            "| Expected | safe | spam | harassment | self_harm_sensitive |",
            "|---|---:|---:|---:|---:|",
            *_confusion_rows(confusion_matrix),
            "",
            "## Latency Summary",
            "",
            "| Component | Count | Avg ms | P50 ms | P95 ms | Max ms |",
            "|---|---:|---:|---:|---:|---:|",
            *_latency_rows(latency),
            "",
            "## Readout",
            "",
            "The current baseline catches the direct rule-matching examples and misses selected",
            "edge cases that require broader semantic coverage. This is intentional for Week 1:",
            "the eval set creates visible room for future model and policy improvements",
            "without adding nondeterministic behavior yet.",
            "",
        ]
    )


def write_eval_report(
    summary: EvalRunSummary,
    report_path: Path = REPORT_PATH,
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_eval_report_markdown(summary), encoding="utf-8")
    return report_path


def _confusion_rows(confusion_matrix: object) -> list[str]:
    rows: list[str] = []
    matrix = confusion_matrix if isinstance(confusion_matrix, dict) else {}
    labels = ["safe", "spam", "harassment", "self_harm_sensitive"]

    for expected in labels:
        predicted_counts = matrix.get(expected, {})
        rows.append(
            "| "
            + expected
            + " | "
            + " | ".join(str(predicted_counts.get(predicted, 0)) for predicted in labels)
            + " |"
        )
    return rows


def _latency_rows(latency: object) -> list[str]:
    rows: list[str] = []
    latency_summaries = latency if isinstance(latency, dict) else {}

    for name in sorted(latency_summaries):
        values = latency_summaries[name]
        rows.append(
            "| "
            + name
            + " | "
            + f"{values['count']} | "
            + f"{values['avg_ms']:.4f} | "
            + f"{values['p50_ms']:.4f} | "
            + f"{values['p95_ms']:.4f} | "
            + f"{values['max_ms']:.4f} |"
        )
    return rows


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
