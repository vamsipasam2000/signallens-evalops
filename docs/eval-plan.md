# Evaluation Plan

Day 4 adds a deterministic offline evaluation loop for the current rule-based workflow.

## Dataset

- Location: `app/data/eval_set.jsonl`
- Size: 60 examples
- Labels: `safe`, `spam`, `harassment`, `self_harm_sensitive`
- Balance: 15 examples per label
- Format: JSONL with `id`, `content`, `expected_risk`, `expected_action`, `policy_area`, and `severity`

The dataset intentionally includes simple edge cases so the rule-based baseline does not produce a meaningless perfect score.

## Metrics

- accuracy
- macro F1
- false positive rate
- false negative rate
- action agreement
- latency summaries: count, average, min, p50, p95, max

## Runner

- Entry point: `app/evaluators/runner.py`
- Command: `python -m app.evaluators.runner`
- Scope: run the eval set through the existing LangGraph workflow and aggregate metrics.

## Current Baseline

The deterministic Day 4 dataset currently scores:

- accuracy: `0.8000`
- macro F1: `0.8111`
- false positive rate: `0.2000`
- false negative rate: `0.2000`
- action agreement: `0.8000`

The eval runner does not require model calls, databases, Docker, or AWS resources.
