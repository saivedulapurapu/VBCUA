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
