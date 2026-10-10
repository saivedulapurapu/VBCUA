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


# --- Theme and layout ---

import tomllib  # noqa: E402

ROOT = FRONTEND.parent


def test_streamlit_theme_is_light_and_toolbar_minimal():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["theme"]["base"] == "light"
    assert config["client"]["toolbarMode"] == "minimal"


def test_analyse_button_names_concept_and_waits_for_audio(backend):
    at = run(query={"concept": "2"})
    button = at.button[0]
    assert button.label == "Analyse explanation of Database Management System (DBMS)"
    assert button.disabled  # no recording yet


def test_no_api_key_form_in_ui(backend):
    at = run()
    labels = [t.label.lower() for t in at.text_input]
    assert not any("api key" in label for label in labels)


def test_add_concept_form_in_sidebar(backend):
    at = run()
    assert [t.label for t in at.sidebar.text_input] == ["Title", "Keywords (comma separated)"]


def test_empty_state_before_any_result(backend):
    assert "No results yet" in all_text(run())


# --- Results view ---

def test_results_view_has_cards_tabs_and_details(backend):
    at = run(session={"current_evaluation": attempt()})
    assert [m.label for m in at.metric] == ["Overall", "Understanding", "Fluency"]
    assert [m.value for m in at.metric] == ["82.2", "74.5", "100"]
    assert [t.label for t in at.tabs] == ["Feedback", "Keywords", "Transcript", "Details"]
    details = at.dataframe[0].value  # first dataframe is the Details table (no history here)
    assert "Cosine similarity" in list(details["Measure"])
    text = all_text(at)
    assert ":green-badge[Strong Understanding]" in text
    assert "1 of 2 key terms mentioned" in text


def test_results_show_microphone_warning(backend):
    at = run(session={"current_evaluation": attempt(recording_warning="Recording is very quiet — check your microphone.")})
    assert any("check your microphone" in w.value for w in at.warning)


def test_results_escape_transcript(backend):
    text = all_text(run(session={"current_evaluation": attempt(transcription="<script>alert(1)</script>")}))
    assert "<script>alert" not in text
    assert "&lt;script&gt;" in text


# --- AI feedback loaded after the scores ---

@pytest.fixture
def feedback_api(backend, monkeypatch):
    state = {"requests": [], "gets": 0, "request_result": None, "get_results": [], "error": None}

    def request_feedback(eval_id):
        state["requests"].append(eval_id)
        if state["error"]:
            raise state["error"]
        return state["request_result"]

    def get_evaluation(eval_id):
        state["gets"] += 1
        return state["get_results"].pop(0)

    monkeypatch.setattr(api, "request_feedback", request_feedback)
    monkeypatch.setattr(api, "get_evaluation", get_evaluation)
    return state


def done(**kw):
    return attempt(feedback="#### Suggestions\nAI says hi", feedback_source="llm", feedback_status="done",
                   llm_model="gemini-3.5-flash-lite", feedback_seconds=4.7, **kw)


def test_pending_feedback_is_requested_and_shown(feedback_api):
    feedback_api["request_result"] = done()
    at = run(session={"current_evaluation": attempt(feedback_source="rule-based", feedback_status="pending", llm_model=None)})
    assert feedback_api["requests"] == [7]
    assert at.session_state["current_evaluation"]["feedback_source"] == "llm"
    assert "AI says hi" in all_text(at)
    assert "rule-based generator" not in all_text(at)


def test_feedback_generating_elsewhere_is_polled_not_requested_again(feedback_api, monkeypatch):
    feedback_api["request_result"] = attempt(feedback_source="rule-based", feedback_status="generating")
    feedback_api["get_results"] = [done()]
    import time as time_module
    monkeypatch.setattr(time_module, "sleep", lambda s: None)
    at = run(session={"current_evaluation": attempt(feedback_source="rule-based", feedback_status="generating")})
    assert feedback_api["requests"] == [7]  # one request; then it waits on GET
    assert feedback_api["gets"] == 1
    assert "AI says hi" in all_text(at)


def test_done_feedback_is_not_requested(feedback_api):
    run(session={"current_evaluation": done()})
    assert feedback_api["requests"] == []


def test_failed_feedback_offers_retry(feedback_api):
    failed = attempt(feedback="Rule-based text", feedback_source="rule-based", feedback_status="failed", llm_model=None)
    at = run(session={"current_evaluation": failed})
    assert feedback_api["requests"] == []
    assert "could not be generated" in all_text(at)
    feedback_api["request_result"] = done()
    retry = next(b for b in at.button if b.label == "Try AI feedback again")
    retry.click().run()
    assert feedback_api["requests"] == [7]
    assert "AI says hi" in all_text(at)


def test_feedback_request_error_is_friendly(feedback_api):
    feedback_api["error"] = api.BackendError("Cannot reach the analysis server.", "ConnectionError")
    at = run(session={"current_evaluation": attempt(feedback_source="rule-based", feedback_status="pending", llm_model=None)})
    assert any("Cannot reach the analysis server" in e.value for e in at.error)
