# Architecture

SignalLens EvalOps is a small trust-safety evaluation service with a FastAPI API, a deterministic LangGraph workflow, optional Langfuse tracing, and an offline evaluation loop.

```text
Client
  -> FastAPI
      -> GET /health
      -> GET /v1/policy
      -> POST /v1/analyze
      -> POST /v1/evals/run
      -> GET /v1/evals/summary

/v1/analyze
  -> LangGraph workflow
      -> normalize_content
      -> retrieve_policy_context
      -> classify_risk
      -> recommend_action
      -> generate_explanation
      -> evaluate_output
  -> optional Langfuse trace + node latency metadata

/v1/evals/run
  -> app/data/eval_set.jsonl
  -> same LangGraph workflow
  -> aggregate metrics
  -> reports/week1_eval_summary.md

Container
  -> python:3.11-slim
  -> non-root appuser
  -> uvicorn app.main:app on 0.0.0.0:8000
  -> /health Docker healthcheck

AWS lightweight deployment
  -> local Docker build
  -> ECR image repository
  -> ECS Fargate task definition
  -> ECS service with public task IP
  -> CloudWatch log group /ecs/signallens-evalops
```

## Current Boundaries

- No model APIs
- No databases
- Docker is used for local and ECS Fargate runs
- Minimal AWS deployment uses ECS Fargate only
- No Kubernetes, Terraform, CI/CD, or advanced infrastructure
- No background workers
- Report output is a deterministic local markdown artifact

## Production Signals

- Typed request and response schemas
- Reusable workflow nodes
- Request-level trace IDs and per-node latency capture
- Offline eval runner with accuracy, macro F1, false positive rate, false negative rate, action agreement, and latency summaries
- Human-readable eval report for recruiter and hiring-manager review
