import numpy as np
import pytest

from backend.app.services import semantic

REFERENCE_VEC = np.array([1.0, 0.0])
KEYWORDS = ["data", "model"]


class StubModel:
    """Returns a fixed transcript vector so tests control the cosine exactly."""

    def __init__(self, transcript_vec):
        self.transcript_vec = np.asarray(transcript_vec, dtype=float)

    def encode(self, texts):
        return [self.transcript_vec, REFERENCE_VEC]


@pytest.fixture
def use_vector(monkeypatch):
    def _use(vec):
        monkeypatch.setattr(semantic, "get_sbert_model", lambda: StubModel(vec))
    return _use


def score(method="v1"):
    return semantic.evaluate_semantic_understanding("some answer", "reference", KEYWORDS, method=method)


def test_unrelated_answer_has_no_floor(use_vector):
    use_vector([0.0, 1.0])  # cos = 0
    assert score()["similarity_score"] == 0.0
    assert score("b0")["similarity_score"] == 50.0  # the old floor


def test_identical_answer_scores_100(use_vector):
    use_vector([1.0, 0.0])
    assert score()["similarity_score"] == 100.0
    assert score("b0")["similarity_score"] == 100.0


def test_opposite_answer_scores_0(use_vector):
    use_vector([-1.0, 0.0])
    assert score()["similarity_score"] == 0.0
    assert score("b0")["similarity_score"] == 0.0


def test_v1_is_cosine_times_100(use_vector):
    use_vector([0.6, 0.8])  # cos = 0.6
    result = score()
    assert result["cosine_similarity"] == pytest.approx(0.6)
    assert result["similarity_score"] == pytest.approx(60.0)
    assert result["scoring_method"] == "v1"


def test_b0_reproduces_old_formula(use_vector):
    use_vector([0.6, 0.8])  # cos = 0.6 -> (0.6 + 1) / 2 * 100 = 80
    result = score("b0")
    assert result["similarity_score"] == pytest.approx(80.0)
    # combined = 70% similarity + 30% keyword coverage (0 keywords mentioned)
    assert result["combined_score"] == pytest.approx(56.0)


def test_empty_transcript_scores_zero():
    result = semantic.evaluate_semantic_understanding("   ", "reference", KEYWORDS)
    assert result["similarity_score"] == 0.0
    assert result["combined_score"] == 0.0


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        semantic.evaluate_semantic_understanding("x", "reference", KEYWORDS, method="nope")
