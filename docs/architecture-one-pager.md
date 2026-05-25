# SignalLens EvalOps Architecture One-Pager

## What It Is

SignalLens EvalOps is a portfolio-ready evaluation and observability service for trust-safety style AI systems. It classifies content risk, recommends an action, traces workflow execution, and runs a small offline eval set to quantify baseline behavior.

## System Flow

```text
API request
  -> FastAPI schema validation
  -> LangGraph workflow
  -> deterministic policy/rule nodes
  -> trace ID + node latency capture
  -> typed API response

Offline eval
  -> 60 labeled JSONL records
  -> same LangGraph workflow
  -> metric aggregation
  -> markdown evaluation report
```

## Components

| Layer | Responsibility |
|---|---|
| FastAPI | Public API boundary, request validation, JSON responses |
| LangGraph | Explicit workflow orchestration and state passing |
| Static policy | Small local trust-safety policy context |
| Langfuse adapter | Optional trace export, local trace IDs when disabled |
| Eval runner | Replays labeled examples through the production workflow |
| Metrics | Accuracy, macro F1, FPR, FNR, action agreement, latency summaries |
| Reports | Recruiter-readable markdown summary of Week 1 performance |
| Docker | Lightweight local production-style FastAPI container |
| AWS ECS | Minimal Fargate deployment path with ECR and CloudWatch logs |

## Why It Matters

The project shows the core mechanics hiring teams look for in AI evaluation and observability work: structured data contracts, repeatable evals, measurable metrics, traceable workflow steps, and clear reporting. The current implementation is intentionally deterministic so future model or retrieval changes can be measured against a stable baseline.

## Explicit Non-Scope For Week 1

- No model API calls
- No databases
- No asynchronous job queue
- No Kubernetes, Terraform, CI/CD, or complex infrastructure
