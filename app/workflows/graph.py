from collections.abc import Callable
from functools import lru_cache
from uuid import uuid4

from langgraph.graph import END, StateGraph

from app.tracing.langfuse_client import (
    WORKFLOW_VERSION,
    finish_node_span,
    node_span,
    request_trace,
    score_eval_outputs,
)
from app.workflows.nodes import (
    classify_risk,
    evaluate_output,
    generate_explanation,
    normalize_content,
    recommend_action,
    retrieve_policy_context,
)
from app.workflows.state import WorkflowState


def _traced_node(name: str, handler: Callable[[WorkflowState], WorkflowState]):
    def wrapped(state: WorkflowState) -> WorkflowState:
        input_payload = {
            "request_id": state.get("request_id"),
            "input_length": state.get("input_length"),
            "risk_category": state.get("risk_category"),
            "recommended_action": state.get("recommended_action"),
        }

        with node_span(name, state, input_payload=input_payload) as timing:
            patch = handler(state)

        latency_ms = finish_node_span(
            timing,
            output_payload=patch,
            metadata={
                "risk_category": patch.get("risk_category", state.get("risk_category")),
                "recommended_action": patch.get(
                    "recommended_action",
                    state.get("recommended_action"),
                ),
                "confidence": patch.get("confidence", state.get("confidence")),
            },
        )

        return {
            **patch,
            "node_latencies_ms": {
                **state.get("node_latencies_ms", {}),
                name: latency_ms,
            },
        }

    return wrapped


@lru_cache
def build_workflow():
    graph = StateGraph(WorkflowState)

    graph.add_node("normalize_content", _traced_node("normalize_content", normalize_content))
    graph.add_node(
        "retrieve_policy_context",
        _traced_node("retrieve_policy_context", retrieve_policy_context),
    )
    graph.add_node("classify_risk", _traced_node("classify_risk", classify_risk))
    graph.add_node("recommend_action", _traced_node("recommend_action", recommend_action))
    graph.add_node(
        "generate_explanation",
        _traced_node("generate_explanation", generate_explanation),
    )
    graph.add_node("evaluate_output", _traced_node("evaluate_output", evaluate_output))

    graph.set_entry_point("normalize_content")
    graph.add_edge("normalize_content", "retrieve_policy_context")
    graph.add_edge("retrieve_policy_context", "classify_risk")
    graph.add_edge("classify_risk", "recommend_action")
    graph.add_edge("recommend_action", "generate_explanation")
    graph.add_edge("generate_explanation", "evaluate_output")
    graph.add_edge("evaluate_output", END)

    return graph.compile()


def run_workflow(initial_state: WorkflowState) -> WorkflowState:
    request_id = initial_state.get("request_id") or str(uuid4())
    raw_content = initial_state["raw_content"]
    input_length = len(raw_content)
    metadata = {
        "request_id": request_id,
        "source": initial_state.get("source"),
        "workflow_version": WORKFLOW_VERSION,
        "input_length": input_length,
        **initial_state.get("metadata", {}),
    }

    with request_trace(
        input_payload={"content": raw_content, "source": initial_state.get("source")},
        metadata=metadata,
    ) as active_trace:
        result = build_workflow().invoke(
            {
                **initial_state,
                "request_id": request_id,
                "trace_id": active_trace.trace_id,
                "tracing_enabled": active_trace.enabled,
                "input_length": input_length,
                "node_latencies_ms": {},
            }
        )

        output = {
            "risk_category": result["risk_category"],
            "recommended_action": result["recommended_action"],
            "confidence": result["confidence"],
            "eval_scores": result["eval_scores"],
            "node_latencies_ms": result["node_latencies_ms"],
        }
        active_trace.update_output(
            output=output,
            metadata={
                **metadata,
                "risk_category": result["risk_category"],
                "recommended_action": result["recommended_action"],
                "confidence": result["confidence"],
            },
        )
        score_eval_outputs(active_trace, result["eval_scores"])

    return result
