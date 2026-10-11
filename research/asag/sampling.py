"""
Stratified sampling and the dev/test split, fixed before any scoring is run.

Strata are (question_id, label). The sample is split across labels in proportion to their
size, then each label's share across questions (largest-remainder rounding, ties broken by the
seeded generator). With ~300 strata and 300 answers most (question, label) cells get 0 or 1
answers, but the label mix matches the dataset and questions are covered in proportion.
"""
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

SEED = 42
N_SAMPLE = 300
N_DEV = 100
STRATA = ("question_id", "label")


def _allocate(sizes: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Splits n proportionally to sizes: floors, then largest remainders (random tie-break)."""
    quotas = sizes / sizes.sum() * n
    counts = np.floor(quotas).astype(int)
    order = np.lexsort((rng.random(len(sizes)), -(quotas - counts)))
    counts[order[: n - counts.sum()]] += 1
    return counts


def stratified_ids(df: pd.DataFrame, n: int, seed: int = SEED, strata: Sequence[str] = STRATA) -> List[str]:
    """
    Returns n item_ids, deterministic for a seed. Allocation is nested from the last stratum
    column inwards: first n is split across labels (so label shares match the data exactly up
    to rounding), then each label's share across questions, then answers are drawn at random.
    """
    if n > len(df):
        raise ValueError(f"Asked for {n} items but the dataset has {len(df)}")
    rng = np.random.default_rng(seed)
    chosen: List[str] = []

    def draw(part: pd.DataFrame, k: int, columns: List[str]) -> None:
        if k == 0:
            return
        if not columns:
            chosen.extend(rng.choice(sorted(part["item_id"]), size=k, replace=False).tolist())
            return
        groups = [g for _, g in part.groupby(columns[-1], sort=True)]
        counts = _allocate(np.array([len(g) for g in groups], dtype=float), k, rng)
        for group, count in zip(groups, counts):
            draw(group, int(count), columns[:-1])

    draw(df, n, list(strata))
    return sorted(chosen)


def make_split(df: pd.DataFrame, n_sample: int = N_SAMPLE, n_dev: int = N_DEV, seed: int = SEED) -> pd.DataFrame:
    """
    Stratified sample of n_sample answers, then a stratified dev subset of n_dev; the rest is test.
    Returns [item_id, question_id, split] sorted by item_id. No text or labels, so it can be committed.
    """
    sample_ids = stratified_ids(df, n_sample, seed)
    sample = df[df["item_id"].isin(sample_ids)]
    dev_ids = set(stratified_ids(sample, n_dev, seed + 1))
    split = sample[["item_id", "question_id"]].copy()
    split["split"] = ["dev" if i in dev_ids else "test" for i in split["item_id"]]
    return split.sort_values("item_id").reset_index(drop=True)


def save_split(split: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(path, index=False)


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def apply_split(df: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """The sampled rows of df with a `split` column; fails if any saved ID is not in the dataset."""
    missing = set(split["item_id"]) - set(df["item_id"])
    if missing:
        raise ValueError(f"{len(missing)} split IDs are not in the dataset (different release?), e.g. {sorted(missing)[:3]}")
    return df.merge(split[["item_id", "split"]], on="item_id", how="inner")
