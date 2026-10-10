import pytest

from backend.app.config import settings
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


def llm_feedback():
    return evaluation.generate_llm_feedback("Machine Learning", "reference", "transcript", SEMANTIC, FLUENCY)


def rule_based():
    return evaluation.generate_rule_based_feedback("Machine Learning", SEMANTIC, FLUENCY)


@pytest.fixture
def llm(monkeypatch):
    state = {"configured": True, "calls": [], "fail": False}

    def fake_generate_text(prompt, **kwargs):
        state["calls"].append(kwargs)
        if state["fail"]:
            raise llm_client.LLMError("api down")
        return "LLM feedback"

    monkeypatch.setattr(llm_client, "is_configured", lambda: state["configured"])
    monkeypatch.setattr(llm_client, "generate_text", fake_generate_text)
    monkeypatch.setattr(settings, "GEMINI_FEEDBACK_MODEL", "feedback-model")
    return state


# --- evaluate_explanation: scores + rule-based feedback only, never calls the LLM ---

def test_evaluate_never_calls_llm(llm):
    result = evaluate()
    assert llm["calls"] == []
    assert result["feedback"] == rule_based()
    assert (result["feedback_source"], result["llm_model"]) == ("rule-based", None)


def test_evaluate_marks_ai_feedback_pending_when_configured(llm):
    assert evaluate()["feedback_status"] == evaluation.FEEDBACK_PENDING


def test_evaluate_marks_ai_feedback_unavailable_without_key(llm):
    llm["configured"] = False
    assert evaluate()["feedback_status"] == evaluation.FEEDBACK_UNAVAILABLE


def test_scores_unchanged_by_feedback_split(llm):
    result = evaluate()
    assert result["final_score"] == pytest.approx(0.7 * 60.0 + 0.3 * 80.0)
    assert result["understanding_grade"] == "Moderate Understanding"


# --- generate_llm_feedback ---

def test_llm_feedback_uses_feedback_model_and_times_call(llm):
    result = llm_feedback()
    assert result["feedback"] == "LLM feedback"
    assert result["llm_model"] == "feedback-model"
    assert result["feedback_seconds"] >= 0
    assert llm["calls"] == [{"model": "feedback-model"}]


def test_llm_feedback_failure_raises_with_elapsed_time(llm):
    llm["fail"] = True
    with pytest.raises(llm_client.LLMError) as info:
        llm_feedback()
    assert info.value.feedback_seconds >= 0


def test_default_feedback_model_is_flash_lite_and_grader_model_unchanged():
    fields = type(settings).model_fields
    assert fields["GEMINI_FEEDBACK_MODEL"].default == "gemini-3.5-flash-lite"
    assert fields["GEMINI_MODEL"].default == "gemini-3.8-flash"
