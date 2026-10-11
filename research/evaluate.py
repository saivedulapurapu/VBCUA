"""
Agreement between automatic scores and human labels, with bootstrap confidence intervals.

Metrics (each with a 95% percentile bootstrap CI, 1000 resamples, fixed seed):
- QWK: Cohen's kappa with quadratic weights over the ordered labels
- Spearman's rho: rank correlation (uses a method's continuous score column if present)
- Accuracy: exact label agreement
plus a confusion matrix per method (rows = human, columns = method).

Usage:
    python research/evaluate.py predictions.csv --labels incorrect partial correct \\
        --methods v1_cosine b0_cosine rubric --out research/results/engsaf_test

The CSV needs a `human` column with labels and one label column per method. If a column
`<method>_score` exists, Spearman uses it instead of the method's labels.
"""
import argparse
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

N_RESAMPLES = 1000
SEED = 42
CI_LEVEL = 0.95


def to_ordinal(values: Sequence, labels: Sequence[str]) -> np.ndarray:
    """Maps labels to 0..k-1 in the given order; raises on an unknown label."""
    index = {label: i for i, label in enumerate(labels)}
    unknown = sorted({str(v) for v in values if v not in index})
    if unknown:
        raise ValueError(f"Labels not in --labels: {unknown}")
    return np.array([index[v] for v in values], dtype=float)


_WEIGHTS: Dict[int, np.ndarray] = {}


def qwk(y_true: np.ndarray, y_pred: np.ndarray, n_labels: int) -> float:
    """
    Quadratic weighted kappa; NaN when undefined (e.g. both raters use a single label).
    Same result as sklearn's cohen_kappa_score(weights="quadratic"), computed directly with
    NumPy because the bootstrap calls it thousands of times.
    """
    t = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    if len(t) == 0:
        return math.nan
    observed = np.bincount(t * n_labels + p, minlength=n_labels * n_labels).reshape(n_labels, n_labels)
    expected = np.outer(observed.sum(1), observed.sum(0)) / len(t)
    if n_labels not in _WEIGHTS:
        i, j = np.indices((n_labels, n_labels))
        _WEIGHTS[n_labels] = (i - j) ** 2
    weights = _WEIGHTS[n_labels]
    denominator = (weights * expected).sum()
    if denominator == 0:
        return math.nan
    return float(1 - (weights * observed).sum() / denominator)


def spearman(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Spearman's rho; NaN when either side is constant."""
    if np.ptp(y_true) == 0 or np.ptp(scores) == 0:
        return math.nan
    return float(spearmanr(y_true, scores).statistic)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred)) if len(y_true) else math.nan


def bootstrap_ci(
    metric: Callable[[np.ndarray], float],
    n_items: int,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    level: float = CI_LEVEL,
) -> tuple:
    """
    Percentile bootstrap CI for metric(indices). Resamples item indices with replacement;
    resamples where the metric is undefined (NaN) are skipped.
    Returns (low, high, n_valid_resamples).
    """
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n_items, size=n_items)
        value = metric(idx)
        if not math.isnan(value):
            values.append(value)
    if not values:
        return math.nan, math.nan, 0
    alpha = (1 - level) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha)), len(values)


def confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> pd.DataFrame:
    k = len(labels)
    matrix = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        matrix[t, p] += 1
    return pd.DataFrame(matrix, index=[f"human: {l}" for l in labels], columns=[f"pred: {l}" for l in labels])


def evaluate_method(
    human: Sequence, predicted: Sequence, labels: Sequence[str], scores: Optional[Sequence[float]] = None,
    n_resamples: int = N_RESAMPLES, seed: int = SEED,
) -> Dict[str, float]:
    """Point estimates and bootstrap CIs for one method. Same resamples for every metric (same seed)."""
    y_true = to_ordinal(human, labels)
    y_pred = to_ordinal(predicted, labels)
    s = np.asarray(scores, dtype=float) if scores is not None else y_pred
    if len(y_true) != len(y_pred) or len(s) != len(y_true):
        raise ValueError("human, predictions and scores must have the same length")
    k = len(labels)
    metrics = {
        "qwk": (lambda i: qwk(y_true[i], y_pred[i], k)),
        "spearman": (lambda i: spearman(y_true[i], s[i])),
        "accuracy": (lambda i: accuracy(y_true[i], y_pred[i])),
    }
    everyone = np.arange(len(y_true))
    result: Dict[str, float] = {"n": len(y_true)}
    for name, fn in metrics.items():
        low, high, valid = bootstrap_ci(fn, len(y_true), n_resamples, seed)
        result[name] = fn(everyone)
        result[f"{name}_ci_low"] = low
        result[f"{name}_ci_high"] = high
        result[f"{name}_ci_resamples"] = valid
    return result


def evaluate_frame(df: pd.DataFrame, methods: List[str], labels: Sequence[str], human_col: str = "human") -> pd.DataFrame:
    rows = []
    for method in methods:
        score_col = f"{method}_score"
        scores = df[score_col].to_numpy(dtype=float) if score_col in df.columns else None
        result = evaluate_method(df[human_col].tolist(), df[method].tolist(), labels, scores)
        rows.append({"method": method, "spearman_on": score_col if scores is not None else method, **result})
    return pd.DataFrame(rows)


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def metrics_markdown(table: pd.DataFrame, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"95% percentile bootstrap CIs, {N_RESAMPLES} resamples, seed {SEED}.",
        "",
        "| Method | n | QWK [95% CI] | Spearman [95% CI] | Accuracy [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        cells = [
            f"{_fmt(r[m])} [{_fmt(r[m + '_ci_low'])}, {_fmt(r[m + '_ci_high'])}]" for m in ("qwk", "spearman", "accuracy")
        ]
        lines.append(f"| {r['method']} | {int(r['n'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def confusion_markdown(matrix: pd.DataFrame, method: str) -> str:
    header = "| | " + " | ".join(matrix.columns) + " |"
    lines = [f"## Confusion matrix: {method}", "", header, "|---" * (len(matrix.columns) + 1) + "|"]
    for label, row in matrix.iterrows():
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def write_report(df: pd.DataFrame, methods: List[str], labels: Sequence[str], out_dir: Path, title: str, human_col: str = "human") -> pd.DataFrame:
    """Writes metrics.csv / metrics.md and confusion_<method>.csv per method; returns the metrics table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table = evaluate_frame(df, methods, labels, human_col)
    table.to_csv(out_dir / "metrics.csv", index=False)
    md = [metrics_markdown(table, title)]
    y_true = to_ordinal(df[human_col].tolist(), labels)
    for method in methods:
        matrix = confusion(y_true, to_ordinal(df[method].tolist(), labels), labels)
        matrix.to_csv(out_dir / f"confusion_{method}.csv")
        md.append(confusion_markdown(matrix, method))
    (out_dir / "metrics.md").write_text("\n".join(md), encoding="utf-8")
    return table


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Agreement metrics with bootstrap CIs.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--labels", nargs="+", required=True, help="labels from lowest to highest")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--human", default="human")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Agreement with human labels")
    args = parser.parse_args(argv)
    table = write_report(pd.read_csv(args.csv), args.methods, args.labels, args.out, args.title, args.human)
    print(metrics_markdown(table, args.title))


if __name__ == "__main__":
    main()
