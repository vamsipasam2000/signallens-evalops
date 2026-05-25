from collections import Counter

from app.evaluators.types import EvalPrediction, EvalRecord, LatencySummary
from app.workflows.state import RiskCategory

RISK_LABELS: tuple[RiskCategory, ...] = (
    "safe",
    "spam",
    "harassment",
    "self_harm_sensitive",
)


def accuracy(records: list[EvalRecord], predictions: list[EvalPrediction]) -> float:
    _validate_pairs(records, predictions)
    correct = sum(
        record.expected_risk == prediction.risk_category
        for record, prediction in zip(records, predictions, strict=True)
    )
    return _round(correct / len(records))


def action_agreement(records: list[EvalRecord], predictions: list[EvalPrediction]) -> float:
    _validate_pairs(records, predictions)
    correct = sum(
        record.expected_action == prediction.recommended_action
        for record, prediction in zip(records, predictions, strict=True)
    )
    return _round(correct / len(records))


def macro_f1(records: list[EvalRecord], predictions: list[EvalPrediction]) -> float:
    _validate_pairs(records, predictions)
    true_labels = [record.expected_risk for record in records]
    pred_labels = [prediction.risk_category for prediction in predictions]
    per_label_f1: list[float] = []

    for label in RISK_LABELS:
        true_positive = sum(
            true_label == label and pred_label == label
            for true_label, pred_label in zip(true_labels, pred_labels, strict=True)
        )
        false_positive = sum(
            true_label != label and pred_label == label
            for true_label, pred_label in zip(true_labels, pred_labels, strict=True)
        )
        false_negative = sum(
            true_label == label and pred_label != label
            for true_label, pred_label in zip(true_labels, pred_labels, strict=True)
        )

        precision = _safe_div(true_positive, true_positive + false_positive)
        recall = _safe_div(true_positive, true_positive + false_negative)
        per_label_f1.append(_safe_div(2 * precision * recall, precision + recall))

    return _round(sum(per_label_f1) / len(per_label_f1))


def false_positive_rate(records: list[EvalRecord], predictions: list[EvalPrediction]) -> float:
    _validate_pairs(records, predictions)
    safe_count = sum(record.expected_risk == "safe" for record in records)
    false_positives = sum(
        record.expected_risk == "safe" and prediction.risk_category != "safe"
        for record, prediction in zip(records, predictions, strict=True)
    )
    return _round(_safe_div(false_positives, safe_count))


def false_negative_rate(records: list[EvalRecord], predictions: list[EvalPrediction]) -> float:
    _validate_pairs(records, predictions)
    unsafe_count = sum(record.expected_risk != "safe" for record in records)
    false_negatives = sum(
        record.expected_risk != "safe" and prediction.risk_category == "safe"
        for record, prediction in zip(records, predictions, strict=True)
    )
    return _round(_safe_div(false_negatives, unsafe_count))


def latency_summaries(predictions: list[EvalPrediction]) -> dict[str, LatencySummary]:
    if not predictions:
        raise ValueError("At least one prediction is required to summarize latency.")

    total_latencies = [prediction.total_latency_ms for prediction in predictions]
    summaries = {"total": summarize_latencies(total_latencies)}

    node_names = sorted(
        {
            node_name
            for prediction in predictions
            for node_name in prediction.node_latencies_ms
        }
    )
    for node_name in node_names:
        summaries[node_name] = summarize_latencies(
            [prediction.node_latencies_ms.get(node_name, 0.0) for prediction in predictions]
        )

    return summaries


def confusion_counts(
    records: list[EvalRecord],
    predictions: list[EvalPrediction],
) -> dict[str, dict[str, int]]:
    _validate_pairs(records, predictions)
    counts: dict[str, Counter[str]] = {label: Counter() for label in RISK_LABELS}
    for record, prediction in zip(records, predictions, strict=True):
        counts[record.expected_risk][prediction.risk_category] += 1
    return {label: dict(counts[label]) for label in RISK_LABELS}


def summarize_latencies(values: list[float]) -> LatencySummary:
    if not values:
        raise ValueError("At least one latency value is required.")

    sorted_values = sorted(values)
    return LatencySummary(
        count=len(sorted_values),
        avg_ms=_round(sum(sorted_values) / len(sorted_values)),
        min_ms=_round(sorted_values[0]),
        p50_ms=_round(_percentile(sorted_values, 50)),
        p95_ms=_round(_percentile(sorted_values, 95)),
        max_ms=_round(sorted_values[-1]),
    )


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile / 100)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _validate_pairs(records: list[EvalRecord], predictions: list[EvalPrediction]) -> None:
    if not records:
        raise ValueError("At least one eval record is required.")
    if len(records) != len(predictions):
        raise ValueError("Eval records and predictions must have the same length.")


def _round(value: float) -> float:
    return round(value, 4)
