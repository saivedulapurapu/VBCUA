import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("evaluate", ROOT / "research" / "evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

LABELS = ["incorrect", "partial", "correct"]


def test_qwk_known_values():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert ev.qwk(y, y, 3) == pytest.approx(1.0)
    # Hand-computed with weights (i-j)^2/4: observed weighted disagreement 0.25, expected 0.75
    # -> QWK = 1 - 0.25 / 0.75 = 0.667
    assert ev.qwk(np.array([0, 1, 2]), np.array([1, 1, 2]), 3) == pytest.approx(2 / 3)
    # Completely reversed ordering is worse than chance
    assert ev.qwk(np.array([0, 0, 2, 2]), np.array([2, 2, 0, 0]), 3) < 0


def test_qwk_matches_reference_formula():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 3, 50), rng.integers(0, 3, 50)
    k = 3
    observed = np.zeros((k, k))
    for i, j in zip(a, b):
        observed[i, j] += 1
    weights = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    expected = np.outer(observed.sum(1), observed.sum(0)) / observed.sum()
    reference = 1 - (weights * observed).sum() / (weights * expected).sum()
    assert ev.qwk(a, b, k) == pytest.approx(reference)


def test_qwk_matches_sklearn():
    cohen_kappa_score = pytest.importorskip("sklearn.metrics").cohen_kappa_score
    rng = np.random.default_rng(5)
    for k in (3, 6):
        a, b = rng.integers(0, k, 80), rng.integers(0, k, 80)
        expected = cohen_kappa_score(a, b, labels=list(range(k)), weights="quadratic")
        assert ev.qwk(a, b, k) == pytest.approx(expected)


def test_undefined_metrics_are_nan():
    constant = np.array([1, 1, 1])
    assert math.isnan(ev.qwk(constant, constant, 3))
    assert math.isnan(ev.spearman(constant, np.array([0.1, 0.5, 0.9])))


def test_spearman_and_accuracy():
    assert ev.spearman(np.array([0, 1, 2, 2]), np.array([0.1, 0.4, 0.8, 0.9])) == pytest.approx(0.9487, abs=1e-3)
    assert ev.accuracy(np.array([0, 1, 2, 2]), np.array([0, 1, 1, 2])) == 0.75


def test_unknown_label_rejected():
    with pytest.raises(ValueError):
        ev.to_ordinal(["correct", "wrong"], LABELS)


def test_bootstrap_is_reproducible_and_brackets_estimate():
    rng = np.random.default_rng(1)
    human = rng.integers(0, 3, 200)
    pred = np.where(rng.random(200) < 0.8, human, rng.integers(0, 3, 200))
    names = [LABELS[i] for i in human], [LABELS[i] for i in pred]
    first = ev.evaluate_method(*names, LABELS)
    second = ev.evaluate_method(*names, LABELS)
    assert first == second  # fixed seed
    for metric in ("qwk", "spearman", "accuracy"):
        assert first[f"{metric}_ci_low"] <= first[metric] <= first[f"{metric}_ci_high"]
        assert first[f"{metric}_ci_resamples"] == ev.N_RESAMPLES
    assert ev.N_RESAMPLES == 1000 and ev.CI_LEVEL == 0.95


def test_spearman_uses_continuous_scores_when_given():
    human = ["incorrect", "partial", "correct", "correct"]
    pred = ["partial", "partial", "partial", "partial"]  # constant labels
    labels_only = ev.evaluate_method(human, pred, LABELS)
    with_scores = ev.evaluate_method(human, pred, LABELS, scores=[0.1, 0.4, 0.8, 0.9])
    assert math.isnan(labels_only["spearman"])
    assert with_scores["spearman"] == pytest.approx(0.9487, abs=1e-3)


def test_confusion_matrix():
    m = ev.confusion(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 0]), LABELS)
    assert m.loc["human: incorrect", "pred: incorrect"] == 1
    assert m.loc["human: incorrect", "pred: partial"] == 1
    assert m.loc["human: correct", "pred: incorrect"] == 1
    assert int(m.values.sum()) == 4


def test_cli_writes_csv_and_markdown(tmp_path):
    df = pd.DataFrame({
        "id": range(6),
        "human": ["incorrect", "partial", "correct", "correct", "partial", "incorrect"],
        "rubric": ["incorrect", "partial", "correct", "partial", "partial", "incorrect"],
        "cosine": ["partial", "partial", "partial", "correct", "incorrect", "partial"],
        "cosine_score": [0.4, 0.5, 0.6, 0.9, 0.2, 0.45],
    })
    csv = tmp_path / "preds.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "results"
    ev.main([str(csv), "--labels", *LABELS, "--methods", "rubric", "cosine", "--out", str(out), "--title", "Pilot"])
    metrics = pd.read_csv(out / "metrics.csv")
    assert list(metrics["method"]) == ["rubric", "cosine"]
    assert metrics.loc[1, "spearman_on"] == "cosine_score"
    md = (out / "metrics.md").read_text(encoding="utf-8")
    assert md.startswith("# Pilot") and "| rubric | 6 |" in md and "Confusion matrix: cosine" in md
    assert (out / "confusion_rubric.csv").exists() and (out / "confusion_cosine.csv").exists()


def test_pearson_and_rmse():
    assert ev.pearson(np.array([1.0, 2, 3]), np.array([2.0, 4, 6])) == pytest.approx(1.0)
    assert math.isnan(ev.pearson(np.array([1.0, 1, 1]), np.array([1.0, 2, 3])))
    assert ev.rmse(np.array([0.0, 0, 0, 0]), np.array([1.0, 1, -1, -1])) == 1.0


def test_evaluate_continuous_has_cis():
    rng = np.random.default_rng(2)
    gold = rng.uniform(0, 5, 150)
    pred = gold + rng.normal(0, 0.5, 150)
    result = ev.evaluate_continuous(gold, pred)
    assert result == ev.evaluate_continuous(gold, pred)
    for metric in ("pearson", "spearman", "rmse"):
        assert result[f"{metric}_ci_low"] <= result[metric] <= result[f"{metric}_ci_high"]
    assert 0.4 < result["rmse"] < 0.6 and result["pearson"] > 0.9
    table = pd.DataFrame([{"method": "m", **result}])
    assert "| m | 150 |" in ev.continuous_markdown(table, "T", "0-5")


def test_spearman_matches_scipy_with_ties():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(9)
    a, b = rng.integers(0, 3, 60).astype(float), rng.normal(size=60).round(1)
    assert ev.spearman(a, b) == pytest.approx(spearmanr(a, b).statistic)
