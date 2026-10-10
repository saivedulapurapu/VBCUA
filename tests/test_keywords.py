import numpy as np
import pytest

from backend.app.services import semantic
from backend.app.services.semantic import match_keywords


def found(transcript, keyword, method="v1"):
    mentioned, _ = match_keywords(transcript, [keyword], method)
    return mentioned == [keyword]


@pytest.mark.parametrize("transcript, keyword", [
    ("we store it in a database", "data"),
    ("i use mysql at work", "sql"),
    ("the model needs retraining", "training"),
    ("on demand", "demand-driven"),
])
def test_v1_no_substring_matches(transcript, keyword):
    assert not found(transcript, keyword)


@pytest.mark.parametrize("transcript, keyword", [
    ("Data is collected first.", "data"),
    ("We write a query in (SQL).", "sql"),
    ("SQL is declarative", "Sql"),
    ("it uses neural networks to learn", "neural networks"),
    ("i write c++ daily", "C++"),
])
def test_v1_whole_word_case_insensitive(transcript, keyword):
    assert found(transcript, keyword)


def test_v1_symbol_keyword_not_matched_by_prefix():
    assert not found("i write c daily", "C++")


@pytest.mark.parametrize("transcript", ["on-demand services", "on demand services", "on  demand services"])
def test_v1_hyphen_and_space_interchangeable(transcript):
    assert found(transcript, "on-demand")


def test_v1_hyphen_keyword_needs_all_words():
    assert not found("based on user demand", "on-demand")
    assert not found("demand grows", "on-demand")


@pytest.mark.parametrize("transcript, keyword", [
    ("it is an algorithm", "algorithms"),
    ("many algorithms exist", "algorithm"),
    ("a relational database", "databases"),
    ("rows live in tables", "table"),
    ("each class has methods", "classes"),
    ("several classes", "class"),
    ("cloud servers", "server"),
])
def test_v1_simple_plurals(transcript, keyword):
    assert found(transcript, keyword)


def test_v1_plural_only_on_last_word():
    assert found("deep neural network", "neural networks")
    assert not found("neurals networks", "neural networks")


@pytest.mark.parametrize("transcript, keyword", [
    ("we store it in a database", "data"),
    ("i use mysql at work", "sql"),
])
def test_b0_keeps_substring_fallback(transcript, keyword):
    assert found(transcript, keyword, method="b0")


def test_unknown_keyword_method_rejected():
    with pytest.raises(ValueError):
        match_keywords("x", ["y"], method="nope")


class StubModel:
    def encode(self, texts):
        return [np.array([0.6, 0.8]), np.array([1.0, 0.0])]  # cos = 0.6


@pytest.mark.parametrize("method, coverage, combined", [
    # v1: only "sql" matches -> 50% coverage; 0.7 * 60 + 0.3 * 50 = 57
    ("v1", 50.0, 57.0),
    # b0: "data" also matches inside "database" -> 100%; 0.7 * 80 + 0.3 * 100 = 86
    ("b0", 100.0, 86.0),
])
def test_coverage_and_combined_score(monkeypatch, method, coverage, combined):
    monkeypatch.setattr(semantic, "get_sbert_model", lambda: StubModel())
    result = semantic.evaluate_semantic_understanding(
        "a database queried with SQL", "reference", ["data", "sql"], method=method
    )
    assert result["keyword_coverage"] == coverage
    assert result["combined_score"] == pytest.approx(combined)
