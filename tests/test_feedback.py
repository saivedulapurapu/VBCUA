import pytest

from backend.app.services import evaluation, llm_client

SEMANTIC = {
    "similarity_score": 60.0,
    "combined_score": 60.0,
    "mentioned_keywords": ["data"],
    "missing_keywords": ["model"],
}
FLUENCY = {"fluency_score": 80.0, "deductions": {}, "observations": ["Natural pacing."]}


def evaluate():
    return evaluation.evaluate_explanation(
        concept_title="Machine Learning",
        reference_description="reference",
        user_transcript="transcript",
        semantic_result=SEMANTIC,
        fluency_result=FLUENCY,
    )


def rule_based():
    return evaluation.generate_rule_based_feedback("Machine Learning", SEMANTIC, FLUENCY)


@pytest.fixture
def llm(monkeypatch):
    state = {"configured": True, "calls": 0, "fail": False}

    def fake_generate_text(prompt, **kwargs):
        state["calls"] += 1
        if state["fail"]:
            raise llm_client.LLMError("api down")
        return "LLM feedback"

    monkeypatch.setattr(llm_client, "is_configured", lambda: state["configured"])
    monkeypatch.setattr(llm_client, "generate_text", fake_generate_text)
    return state


def test_no_key_uses_rule_based_without_calling_llm(llm):
    llm["configured"] = False
    assert evaluate()["feedback"] == rule_based()
    assert llm["calls"] == 0


def test_llm_failure_falls_back_to_rule_based(llm):
    llm["fail"] = True
    assert evaluate()["feedback"] == rule_based()
    assert llm["calls"] == 1


def test_llm_feedback_used_when_available(llm):
    assert evaluate()["feedback"] == "LLM feedback"
