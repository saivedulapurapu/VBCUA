"""The six draft concept rubrics and the generated review document."""
import pytest

from research.asag import concept_rubrics as cr

EXPECTED = {"normalisation", "deadlock", "tcp-vs-udp", "overfitting", "virtual-memory", "process-vs-thread"}


@pytest.fixture(scope="module")
def rubrics():
    return cr.load_all()


def test_all_six_concepts_present(rubrics):
    assert {r["id"] for r in rubrics} == EXPECTED


def test_every_rubric_is_marked_draft(rubrics):
    for r in rubrics:
        assert r["status"] == cr.DRAFT


@pytest.mark.parametrize("index", range(6))
def test_rubric_shape(rubrics, index):
    r = rubrics[index]
    assert 4 <= len(r["key_points"]) <= 6
    assert 2 <= len(r["misconceptions"]) <= 3
    assert [k["id"] for k in r["key_points"]] == [f"K{i}" for i in range(1, len(r["key_points"]) + 1)]
    assert [m["id"] for m in r["misconceptions"]] == [f"M{i}" for i in range(1, len(r["misconceptions"]) + 1)]
    assert set(r["scale"]) == {"0", "1", "2"}
    assert all(k["text"].strip() and k["partial_if"].strip() for k in r["key_points"])


def test_review_document_is_up_to_date(rubrics):
    """research/rubrics/DRAFT_rubrics_for_review.md must be regenerated after any JSON change."""
    assert cr.REVIEW_FILE.read_text(encoding="utf-8") == cr.render_markdown(rubrics)


def test_review_document_marks_draft_everywhere(rubrics):
    md = cr.render_markdown(rubrics)
    assert md.startswith(f"# Concept rubrics for VBCUA — {cr.DRAFT}")
    assert md.count(cr.DRAFT) >= 1 + len(rubrics)
    for r in rubrics:
        assert f"## {r['concept']}" in md
