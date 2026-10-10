import pytest

from frontend.display import FOUND, MISSING, NOT_RECORDED, concept_title, fluency_text, keyword_badges

CLOUD = {"id": 2, "title": "Cloud Computing", "keywords": ["servers", "storage", "on-demand"]}


def attempt(**overrides):
    values = {
        "concept_id": 2,
        "concept": CLOUD,
        "transcription": "We rent servers and storage on demand.",
        "fluency_score": 90.0,
        "filler_words_ratio": 10.0,
        "mentioned_keywords": ["servers", "storage", "on-demand"],
        "missing_keywords": [],
    }
    values.update(overrides)
    return values


def test_fluency_shows_stored_backend_score():
    # filler ratio 10 would give 60% with the old frontend formula
    assert fluency_text(attempt(fluency_score=90.0, filler_words_ratio=10.0)) == "90%"


def test_fluency_not_recomputed_for_old_rows():
    assert fluency_text(attempt(fluency_score=None)) == "n/a"


def test_badges_follow_stored_lists():
    badges = keyword_badges(attempt(mentioned_keywords=["sql"], missing_keywords=["data"]))
    assert badges == [("sql", FOUND), ("data", MISSING)]


def test_data_missing_when_only_database_said():
    # Stored v1 lists for "a database is queried with sql"
    res = attempt(
        transcription="A database is queried with SQL.",
        mentioned_keywords=["sql"], missing_keywords=["data"],
    )
    assert ("data", MISSING) in keyword_badges(res)


def test_badges_use_evaluated_concept_not_sidebar():
    # Whatever the sidebar shows, an older Cloud attempt lists Cloud keywords
    old = attempt(mentioned_keywords=None, missing_keywords=None)
    assert keyword_badges(old) == [(kw, NOT_RECORDED) for kw in CLOUD["keywords"]]


def test_badges_empty_without_concept_or_lists():
    assert keyword_badges(attempt(mentioned_keywords=None, missing_keywords=None, concept=None)) == []


def test_title_from_evaluated_concept():
    assert concept_title(attempt()) == "Cloud Computing"
    assert concept_title(attempt(concept=None, concept_id=7)) == "Concept #7"


# --- Fix 9: every user-provided value is escaped before st.html ---

from frontend.display import (  # noqa: E402
    PLAIN, description_html, esc, keyword_badges_html, transcript_html,
)

HOSTILE = [
    "<script>alert(1)</script>",
    "\"><img src=x onerror=alert(1)>",
    "'><svg onload=alert(1)>",
    "R&D",
]


def assert_inert(output, value):
    assert "<script" not in output
    assert "<img" not in output
    assert "<svg" not in output
    assert esc(value) in output


@pytest.mark.parametrize("value", HOSTILE)
def test_description_escaped(value):
    assert_inert(description_html(value), value)


@pytest.mark.parametrize("value", HOSTILE)
def test_transcript_escaped(value):
    assert_inert(transcript_html(value), value)


@pytest.mark.parametrize("value", HOSTILE)
@pytest.mark.parametrize("status", [FOUND, MISSING, NOT_RECORDED, PLAIN])
def test_keyword_badges_escaped(value, status):
    assert_inert(keyword_badges_html([(value, status)]), value)


def test_esc_escapes_quotes():
    assert esc("a'b\"c") == "a&#x27;b&quot;c"


# --- Navigation helpers ---

from frontend.display import feedback_notice, history_rows, initial_concept_id  # noqa: E402

CONCEPT_LIST = [{"id": 3}, {"id": 5}]


@pytest.mark.parametrize("requested, expected", [("5", 5), (5, 5), ("9", 3), ("abc", 3), (None, 3)])
def test_initial_concept_id(requested, expected):
    assert initial_concept_id(CONCEPT_LIST, requested) == expected


def test_initial_concept_id_empty():
    assert initial_concept_id([], "1") is None


def test_history_rows_newest_first_plain_text():
    items = [
        {"id": 1, "concept": CLOUD, "concept_id": 2, "final_score": 51.456, "understanding_grade": "Moderate Understanding",
         "speech_duration": 79.98, "created_at": "2026-10-10T19:58:00.123"},
        {"id": 2, "concept": {"title": "<b>X</b>"}, "concept_id": 9, "final_score": 76.9, "understanding_grade": "Strong Understanding",
         "speech_duration": 81.9, "created_at": "2026-10-10T20:10:00"},
    ]
    rows = history_rows(items)
    assert [r["id"] for r in rows] == [2, 1]
    assert rows[1] == {"id": 1, "Date": "2026-10-10 19:58", "Concept": "Cloud Computing", "Overall": 51.5,
                       "Grade": "Moderate", "Duration (s)": 80.0}
    assert rows[0]["Concept"] == "<b>X</b>"  # shown as plain text by st.dataframe, never as HTML


def test_feedback_notice_only_for_rule_based():
    assert "rule-based" in feedback_notice({"feedback_source": "rule-based"})
    assert feedback_notice({"feedback_source": "llm"}) is None
    assert feedback_notice({}) is None  # older attempts


# --- Results view helpers ---

from frontend.display import details_rows, grade_badge_color, score_cards  # noqa: E402


@pytest.mark.parametrize("grade, color", [
    ("Strong Understanding", "green"), ("Moderate Understanding", "orange"), ("Poor Understanding", "red"),
])
def test_grade_badge_color(grade, color):
    assert grade_badge_color(grade) == color


def test_score_cards_use_stored_values():
    res = attempt(final_score=76.94, understanding_score=67.0, semantic_score=85.0, keyword_coverage=25.0,
                  fluency_score=100.0, pause_ratio_v1=0.263, pause_ratio=0.5)
    overall, understanding, fluency = score_cards(res)
    assert (overall["label"], overall["value"]) == ("Overall", "76.9")
    assert understanding["value"] == "67.0"
    assert understanding["caption"] == "Similarity 85.0 · keywords 25%"
    assert fluency["value"] == "100"
    assert fluency["caption"] == "Pauses 26% of speaking time"  # v1 pause ratio, not B0


def test_score_cards_old_rows_show_na():
    res = attempt(understanding_score=None, keyword_coverage=None, fluency_score=None, pause_ratio_v1=None,
                  pause_ratio=0.4, final_score=50.0, semantic_score=60.0)
    _, understanding, fluency = score_cards(res)
    assert understanding["value"] == "n/a"
    assert "keywords n/a" in understanding["caption"]
    assert fluency["value"] == "n/a"
    assert fluency["caption"] == "Pauses 40% of speaking time"  # falls back to the stored B0 value


def test_details_rows_cover_all_sub_scores():
    res = attempt(semantic_score=85.0, cosine_similarity=0.8502, keyword_coverage=25.0, understanding_score=67.0,
                  pause_ratio_v1=0.263, final_score=76.9, filler_words_count=0, speech_duration=81.9,
                  asr_model="whisper-small", asr_confidence=-0.2238, feedback_source="rule-based", llm_model=None,
                  semantic_method="v1", fluency_method="v1")
    rows = {r["Measure"]: r for r in details_rows(res)}
    assert rows["Cosine similarity"]["Value"] == "0.850"
    assert rows["Pause ratio"]["Value"] == "26.3%"
    assert rows["Speech-to-text model"]["Value"] == "whisper-small"
    assert rows["Speech-to-text model"]["Note"] == "ASR confidence -0.224"
    assert rows["Feedback source"]["Value"] == "rule-based"
    assert rows["Filler words"]["Note"] == "Experimental, not scored"
    assert rows["Scoring methods"]["Value"] == "semantic v1, fluency v1"


from frontend.display import demote_headings  # noqa: E402


def test_demote_headings():
    md = "# Assessment Report\n## Section\n### Sub\n#### Small\nText with # hash\n#hashtag\n- bullet"
    assert demote_headings(md).splitlines() == [
        "#### Assessment Report", "#### Section", "#### Sub", "#### Small",
        "Text with # hash", "#hashtag", "- bullet",
    ]


from frontend.display import feedback_in_progress  # noqa: E402


@pytest.mark.parametrize("source, status, in_progress", [
    ("rule-based", "pending", True), ("rule-based", "generating", True),
    ("rule-based", "failed", False), ("rule-based", "unavailable", False),
    ("llm", "done", False), ("llm", "pending", False), ("rule-based", None, False),
])
def test_feedback_in_progress(source, status, in_progress):
    assert feedback_in_progress({"feedback_source": source, "feedback_status": status}) is in_progress


def test_feedback_notice_by_status():
    assert feedback_notice({"feedback_source": "rule-based", "feedback_status": "pending"}) is None
    assert "could not be generated" in feedback_notice({"feedback_source": "rule-based", "feedback_status": "failed"})
    assert "No AI key" in feedback_notice({"feedback_source": "rule-based", "feedback_status": "unavailable"})


def test_details_include_feedback_time():
    rows = {r["Measure"]: r for r in details_rows(attempt(feedback_seconds=4.71))}
    assert rows["Feedback time"]["Value"] == "4.7 s"
