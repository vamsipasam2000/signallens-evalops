# SignalLens EvalOps

SignalLens EvalOps is a production-oriented FastAPI project for AI evaluation and observability. It models a trust-safety content triage system with deterministic workflow orchestration, trace-ready node execution, and a small offline eval loop.

## Recruiter Signal

This project demonstrates the work expected in AI evaluation, AI tooling, trust-safety ML, and applied AI systems roles:

- API contracts with FastAPI and Pydantic
- Explicit LangGraph workflow state and node boundaries
- Optional Langfuse tracing with local trace IDs when credentials are absent
- Labeled eval dataset with repeatable metric aggregation
- Markdown eval report suitable for engineering review
- Clear architecture docs and production boundaries

## Current Scope

- `GET /health`
- `GET /v1/policy`
- `POST /v1/analyze`
- `POST /v1/evals/run`
- `GET /v1/evals/summary`
- 60-record JSONL eval dataset
- Metrics: accuracy, macro F1, false positive rate, false negative rate, action agreement, latency summaries
- Report output: `reports/week1_eval_summary.md`
- Architecture overview: `docs/architecture-one-pager.md`
- Lightweight Docker container for local production-style runs
- Minimal AWS ECS Fargate deployment path with ECR and CloudWatch logs

No model calls, databases, Kubernetes, Terraform, CI/CD, or advanced infrastructure are included yet.

## Architecture

```text
FastAPI
  -> LangGraph trust-safety workflow
  -> optional Langfuse tracing
  -> deterministic eval runner
  -> markdown report generation
```

The same workflow powers both live analysis and offline evaluation, so improvements to classification logic are immediately measurable through the eval runner.

## Local Setup

```bash
uv sync
uv run uvicorn app.main:app --reload
```

Without `uv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

## Run Tests

```bash
pytest
```

## Run Offline Eval

```bash
python -m app.evaluators.runner
```

Or through the API:

```bash
curl -X POST http://127.0.0.1:8000/v1/evals/run
curl http://127.0.0.1:8000/v1/evals/summary
```

## Run With Docker

Build the image:

```bash
docker build -t signallens-evalops:local .
```

Run the FastAPI container:

```bash
docker run --rm -p 8000:8000 signallens-evalops:local
```

If port `8000` is already in use locally:

```bash
docker run --rm -p 8001:8000 signallens-evalops:local
```

Smoke-test the container:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/evals/summary
```

The container runs `uvicorn` on `0.0.0.0:8000`, sets `APP_ENV=production`, includes a lightweight healthcheck, and runs as a non-root user.

## Deploy To AWS ECS Fargate

The lightweight AWS path lives in `deploy/aws/`:

- `task-definition.template.json`
- `service.template.json`
- `ecs-task-execution-assume-role-policy.json`
- `README.md`

It covers:

- creating/pushing an image to ECR
- registering an ECS Fargate task definition
- creating an ECS service
- setting runtime environment variables
- shipping container logs to CloudWatch

Start with:

```bash
cat deploy/aws/README.md
```

The AWS path intentionally avoids Kubernetes, Terraform, CI/CD, databases, load balancers, and autoscaling policies.

## Week 1 Baseline

- accuracy: `0.8000`
- macro F1: `0.8111`
- false positive rate: `0.2000`
- false negative rate: `0.2000`
- action agreement: `0.8000`

The dataset includes direct matches and selected edge cases, which makes the baseline useful for measuring future model, retrieval, or policy improvements.

## Optional Langfuse Tracing

Set these values in `.env` to export traces:

```env
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Without credentials, the API still returns a local `trace_id`, node latencies, and eval scores.
