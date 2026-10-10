from datetime import datetime

import pytest

from backend.app.schemas.evaluation import EvaluationResponse
from backend.app.services import evaluation, llm_client
from backend.app.services.audio_analysis import LOW_RMS_THRESHOLD, recording_warning

QUIET = 0.001
NORMAL = 0.05


def test_quiet_recording_not_penalised_in_v1():
    result = evaluation.compute_fluency_score(1.0, 0.15, rms_energy=QUIET)
    assert result["fluency_score"] == 100.0
    assert "speaking_volume" not in result["deductions"]
    assert not any("volume" in obs.lower() for obs in result["observations"])
    assert result["scoring_method"] == "v1"


def test_v1_does_not_need_rms():
    assert evaluation.compute_fluency_score(1.0, 0.15)["fluency_score"] == 100.0


def test_b0_keeps_old_volume_deduction():
    result = evaluation.compute_fluency_score(1.0, 0.15, rms_energy=QUIET, method="b0")
    assert result["fluency_score"] == 85.0
    assert result["deductions"]["speaking_volume"] == 15.0
    assert result["scoring_method"] == "b0"


@pytest.mark.parametrize("filler_ratio, pause_ratio", [(1.0, 0.15), (5.0, 0.40), (3.0, 0.02), (20.0, 0.9)])
def test_v1_matches_b0_for_normal_volume(filler_ratio, pause_ratio):
    v1 = evaluation.compute_fluency_score(filler_ratio, pause_ratio, rms_energy=NORMAL)
    b0 = evaluation.compute_fluency_score(filler_ratio, pause_ratio, rms_energy=NORMAL, method="b0")
    assert v1["fluency_score"] == b0["fluency_score"]
    assert v1["deductions"] == b0["deductions"]


def test_b0_requires_rms():
    with pytest.raises(ValueError):
        evaluation.compute_fluency_score(1.0, 0.15, method="b0")


def test_unknown_fluency_method_rejected():
    with pytest.raises(ValueError):
        evaluation.compute_fluency_score(1.0, 0.15, method="nope")


def test_recording_warning_threshold():
    assert recording_warning(LOW_RMS_THRESHOLD - 0.0001) is not None
    assert "microphone" in recording_warning(QUIET)
    assert recording_warning(LOW_RMS_THRESHOLD) is None
    assert recording_warning(NORMAL) is None
    assert recording_warning(None) is None


def make_response(rms_energy):
    return EvaluationResponse(
        id=1, concept_id=1, transcription="t", semantic_score=50.0,
        filler_words_count=0, filler_words_ratio=0.0, pause_ratio=0.2,
        rms_energy=rms_energy, speech_duration=10.0, final_score=50.0,
        understanding_grade="Moderate Understanding", feedback="f",
        created_at=datetime(2026, 10, 10),
    )


def test_api_response_exposes_recording_warning():
    assert "microphone" in make_response(QUIET).model_dump()["recording_warning"]
    assert make_response(NORMAL).model_dump()["recording_warning"] is None


SEMANTIC = {
    "similarity_score": 60.0,
    "combined_score": 60.0,
    "mentioned_keywords": ["data"],
    "missing_keywords": ["model"],
}


def test_rule_based_feedback_has_no_volume():
    fluency = evaluation.compute_fluency_score(1.0, 0.15, rms_energy=QUIET)
    feedback = evaluation.generate_rule_based_feedback("ML", SEMANTIC, fluency)
    assert "volume" not in feedback.lower()


def test_llm_prompt_has_no_volume(monkeypatch):
    captured = {}

    def fake_generate_text(prompt, **kwargs):
        captured["prompt"] = prompt
        return "ok"

    monkeypatch.setattr(llm_client, "generate_text", fake_generate_text)
    fluency = evaluation.compute_fluency_score(1.0, 0.15, rms_energy=QUIET)
    evaluation.generate_ai_feedback("ML", "reference", "transcript", SEMANTIC, fluency)
    assert "volume" not in captured["prompt"].lower()
