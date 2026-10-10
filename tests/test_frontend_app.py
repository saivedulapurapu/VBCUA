"""
Runs the real Streamlit page with Streamlit's AppTest. The backend API module is
stubbed, so no server, models or network are needed.
"""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
sys.path.insert(0, str(FRONTEND))
import api  # noqa: E402

CONCEPTS = [
    {"id": 1, "title": "Cloud Computing", "description": "Cloud desc", "keywords": ["servers"], "created_at": "2026-10-10T10:00:00"},
    {"id": 2, "title": "Database Management System (DBMS)", "description": "DBMS desc", "keywords": ["sql", "tables"], "created_at": "2026-10-10T10:00:00"},
]


def attempt(**overrides):
    values = {
        "id": 7, "concept_id": 2, "concept": CONCEPTS[1], "transcription": "A DBMS stores data using SQL.",
        "asr_model": "whisper-small", "asr_confidence": -0.22, "semantic_score": 85.0, "cosine_similarity": 0.85,
        "keyword_coverage": 50.0, "mentioned_keywords": ["sql"], "missing_keywords": ["tables"],
        "understanding_score": 74.5, "semantic_method": "v1", "fluency_score": 100.0, "fluency_method": "v1",
        "filler_words_count": 0, "filler_words_ratio": 0.0, "pause_ratio": 0.3, "pause_ratio_v1": 0.26,
        "rms_energy": 0.05, "speech_duration": 82.0, "final_score": 82.2, "understanding_grade": "Strong Understanding",
        "feedback": "Good **work**.", "feedback_source": "llm", "llm_model": "gemini-3.8-flash",
        "audio_path": None, "created_at": "2026-10-10T20:10:00", "recording_warning": None,
    }
    values.update(overrides)
    return values


@pytest.fixture
def backend(monkeypatch):
    state = {"concepts": CONCEPTS, "history": [], "attempts": {}, "concepts_error": None}

    def get_concepts():
        if state["concepts_error"]:
            raise state["concepts_error"]
        return state["concepts"]

    def get_evaluation(eval_id):
        if eval_id not in state["attempts"]:
            raise api.BackendError("The requested item was not found.", "HTTP 404")
        return state["attempts"][eval_id]

    monkeypatch.setattr(api, "get_concepts", get_concepts)
    monkeypatch.setattr(api, "get_evaluations", lambda: state["history"])
    monkeypatch.setattr(api, "get_evaluation", get_evaluation)
    monkeypatch.setattr(api, "get_pdf", lambda eval_id: b"%PDF")
    return state


def run(query=None, session=None):
    at = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=30)
    for key, value in (query or {}).items():
        at.query_params[key] = value
    for key, value in (session or {}).items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def all_text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [e.value for e in at.error] + [i.value for i in at.info] + [w.value for w in at.warning]
    parts += [str(h.proto) for h in at.get("html")]
    return "\n".join(parts)


def test_renders_with_first_concept_by_default(backend):
    at = run()
    assert at.selectbox(key="concept_id").value == 1
    assert at.query_params["concept"] == ["1"] or at.query_params["concept"] == "1"


def test_concept_restored_from_url(backend):
    at = run(query={"concept": "2"})
    assert at.selectbox(key="concept_id").value == 2


def test_concept_choice_kept_across_reruns(backend):
    at = run()
    at.selectbox(key="concept_id").set_value(2).run()
    assert at.selectbox(key="concept_id").value == 2
    at.run()
    assert at.selectbox(key="concept_id").value == 2


def test_unknown_concept_in_url_falls_back_to_first(backend):
    assert run(query={"concept": "99"}).selectbox(key="concept_id").value == 1


def test_backend_down_shows_friendly_error(backend):
    backend["concepts_error"] = api.BackendError("Cannot reach the analysis server.", "ConnectionError: refused")
    at = run()
    assert any("Cannot reach the analysis server" in e.value for e in at.error)
    assert not any("ConnectionError" in e.value for e in at.error)


def test_attempt_restored_from_url_after_reload(backend):
    backend["attempts"][7] = attempt()
    text = all_text(run(query={"attempt": "7", "concept": "1"}))
    # Result is for the evaluated concept (DBMS), even though the sidebar shows Cloud Computing
    assert "Database Management System (DBMS)" in text


def test_missing_attempt_in_url_is_ignored(backend):
    at = run(query={"attempt": "123"})
    assert "attempt" not in at.query_params


def test_rule_based_feedback_notice(backend):
    text = all_text(run(session={"current_evaluation": attempt(feedback_source="rule-based", llm_model=None)}))
    assert "rule-based generator" in text


def test_no_notice_for_llm_feedback(backend):
    assert "rule-based generator" not in all_text(run(session={"current_evaluation": attempt()}))


def test_history_lists_attempts(backend):
    backend["history"] = [attempt(id=1, final_score=51.5), attempt(id=2)]
    at = run()
    assert len(at.dataframe) == 1
    df = at.dataframe[0].value
    assert list(df["Overall"]) == [82.2, 51.5]  # newest first
