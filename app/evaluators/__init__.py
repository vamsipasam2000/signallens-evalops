"""Offline evaluation utilities for deterministic SignalLens workflows."""

from app.evaluators.dataset import load_eval_records
from app.evaluators.metrics import (
    accuracy,
    action_agreement,
    false_negative_rate,
    false_positive_rate,
    latency_summaries,
    macro_f1,
)
from app.evaluators.types import EvalPrediction, EvalRecord, EvalRunSummary, LatencySummary

__all__ = [
    "EvalPrediction",
    "EvalRecord",
    "EvalRunSummary",
    "LatencySummary",
    "accuracy",
    "action_agreement",
    "false_negative_rate",
    "false_positive_rate",
    "latency_summaries",
    "load_eval_records",
    "macro_f1",
]
