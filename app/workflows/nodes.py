from pathlib import Path

from app.workflows.state import RecommendedAction, RiskCategory, WorkflowState

POLICY_PATH = Path(__file__).resolve().parents[1] / "data" / "policy.md"

POLICY_CATEGORIES: list[RiskCategory] = [
    "safe",
    "spam",
    "harassment",
    "self_harm_sensitive",
]

KEYWORD_RULES: dict[RiskCategory, tuple[str, ...]] = {
    "self_harm_sensitive": (
        "kill myself",
        "hurt myself",
        "end it all",
        "not want to be alive",
        "don't want to be alive",
        "suicide",
    ),
    "harassment": (
        "worthless",
        "idiot",
        "stupid",
        "attack your account",
        "hate you",
        "shut up",
    ),
    "spam": (
        "click this link",
        "free followers",
        "guaranteed",
        "buy now",
        "promo code",
        "discount code",
        "http://",
        "https://",
        "www.",
    ),
    "safe": (),
}

ACTION_BY_RISK: dict[RiskCategory, RecommendedAction] = {
    "safe": "allow",
    "spam": "downrank",
    "harassment": "block",
    "self_harm_sensitive": "human_review",
}


def normalize_content(state: WorkflowState) -> WorkflowState:
    normalized = " ".join(state["raw_content"].split())
    return {"normalized_content": normalized}


def retrieve_policy_context(state: WorkflowState) -> WorkflowState:
    policy_text = POLICY_PATH.read_text(encoding="utf-8")
    return {
        "policy_context": policy_text,
        "policy_categories": list(POLICY_CATEGORIES),
    }


def classify_risk(state: WorkflowState) -> WorkflowState:
    content = state["normalized_content"].lower()

    for category in ("self_harm_sensitive", "harassment", "spam"):
        matched_terms = [term for term in KEYWORD_RULES[category] if term in content]
        if matched_terms:
            confidence = min(0.95, 0.72 + (0.06 * len(matched_terms)))
            return {
                "risk_category": category,
                "matched_terms": matched_terms,
                "confidence": round(confidence, 2),
            }

    return {
        "risk_category": "safe",
        "matched_terms": [],
        "confidence": 0.64,
    }


def recommend_action(state: WorkflowState) -> WorkflowState:
    risk_category = state["risk_category"]
    return {"recommended_action": ACTION_BY_RISK[risk_category]}


def generate_explanation(state: WorkflowState) -> WorkflowState:
    risk_category = state["risk_category"]
    action = state["recommended_action"]
    matched_terms = state.get("matched_terms", [])

    if risk_category == "safe":
        explanation = "No policy-risk terms were detected by the Day 2 rule-based workflow."
    else:
        terms = ", ".join(matched_terms)
        explanation = (
            f"Classified as {risk_category} because the content matched: {terms}. "
            f"Recommended action: {action}."
        )

    return {"explanation": explanation}


def evaluate_output(state: WorkflowState) -> WorkflowState:
    has_policy_category = state["risk_category"] in state.get("policy_categories", [])
    has_action = bool(state.get("recommended_action"))
    has_explanation = len(state.get("explanation", "")) >= 20
    confidence = state.get("confidence", 0.0)

    return {
        "eval_scores": {
            "policy_category_valid": has_policy_category,
            "action_present": has_action,
            "explanation_present": has_explanation,
            "confidence": confidence,
            "rule_version": "day2-deterministic-v1",
        }
    }

