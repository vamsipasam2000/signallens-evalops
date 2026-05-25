from fastapi.testclient import TestClient

from app.evaluators.reports import REPORT_PATH
from app.main import app

client = TestClient(app)


def test_health_returns_service_status() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "SignalLens EvalOps"


def test_policy_returns_static_policy() -> None:
    response = client.get("/v1/policy")

    assert response.status_code == 200
    body = response.json()
    assert "harassment" in body["categories"]
    assert "Recommended action" in body["policy"]


def test_analyze_rejects_blank_content() -> None:
    response = client.post("/v1/analyze", json={"content": "   "})

    assert response.status_code == 422


def test_analyze_accepts_valid_content() -> None:
    response = client.post(
        "/v1/analyze",
        json={
            "content": "  This product was helpful and easy to use.  ",
            "source": "unit-test",
            "metadata": {"example_id": "test_001"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"]
    assert body["status"] == "completed"
    assert body["trace_id"]
    assert body["tracing_enabled"] is False
    assert body["normalized_content"] == "This product was helpful and easy to use."
    assert body["risk_category"] == "safe"
    assert body["recommended_action"] == "allow"
    assert body["eval_scores"]["policy_category_valid"] is True
    assert set(body["node_latencies_ms"]) == {
        "normalize_content",
        "retrieve_policy_context",
        "classify_risk",
        "recommend_action",
        "generate_explanation",
        "evaluate_output",
    }


def test_analyze_flags_spam_with_rule_based_workflow() -> None:
    response = client.post(
        "/v1/analyze",
        json={"content": "Click this link now for guaranteed free followers."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["risk_category"] == "spam"
    assert body["recommended_action"] == "downrank"
    assert "click this link" in body["explanation"].lower()
    assert body["eval_scores"]["rule_version"] == "day2-deterministic-v1"


def test_analyze_routes_self_harm_to_human_review() -> None:
    response = client.post(
        "/v1/analyze",
        json={"content": "I do not want to be alive anymore."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["risk_category"] == "self_harm_sensitive"
    assert body["recommended_action"] == "human_review"
    assert body["workflow_version"] == "day3-langgraph-langfuse-deterministic-v1"


def test_eval_summary_endpoint_returns_aggregate_metrics() -> None:
    response = client.get("/v1/evals/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["dataset_size"] == 60
    assert body["dataset_path"] == "app/data/eval_set.jsonl"
    assert body["metrics"]["accuracy"] == 0.8
    assert body["metrics"]["macro_f1"] == 0.8111
    assert body["metrics"]["false_positive_rate"] == 0.2
    assert body["metrics"]["false_negative_rate"] == 0.2
    assert body["metrics"]["action_agreement"] == 0.8
    assert body["latency"]["total"]["count"] == 60
    assert body["confusion_matrix"]["safe"] == {"safe": 12, "spam": 3}


def test_eval_run_endpoint_writes_markdown_report() -> None:
    response = client.post("/v1/evals/run")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["report_path"] == "reports/week1_eval_summary.md"
    assert body["metrics"]["accuracy"] == 0.8
    assert REPORT_PATH.exists()
    assert "## Aggregate Metrics" in REPORT_PATH.read_text(encoding="utf-8")
