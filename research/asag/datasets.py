"""
Short-answer grading datasets in one common shape, so the pilot runs on any of them.

Every loader returns a DataFrame with these columns (one row per student answer):
    item_id      unique, stable ID of the answer
    question_id  ID of the question
    question     question text
    reference    reference (instructor) answer
    answer       student answer
    label        "incorrect" / "partial" / "correct"

Data files stay in research/data/ (git-ignored) and are never committed.
"""
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LABELS = ("incorrect", "partial", "correct")
COLUMNS = ["item_id", "question_id", "question", "reference", "answer", "label"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Checks the common shape; returns the frame with only the common columns, as strings."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns {missing}")
    df = df[COLUMNS].copy()
    for column in COLUMNS:
        df[column] = df[column].astype(str)
    if df["item_id"].duplicated().any():
        raise ValueError("item_id values must be unique")
    bad = sorted(set(df["label"]) - set(LABELS))
    if bad:
        raise ValueError(f"Unknown labels {bad}; expected {list(LABELS)}")
    per_question = df.groupby("question_id")[["question", "reference"]].nunique()
    if (per_question > 1).any().any():
        raise ValueError("Each question_id must have exactly one question text and one reference answer")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- EngSAF
# Column names in the official EngSAF CSVs; override with `columns` if the release differs.
ENGSAF_COLUMNS = {
    "question_id": "Question_id",
    "question": "Question",
    "reference": "Correct Answer",
    "answer": "Student Answer",
    "label": "output_label",
}
# EngSAF output_label: 0 = incorrect, 1 = partially correct, 2 = correct
ENGSAF_LABELS = {"0": "incorrect", "1": "partial", "2": "correct"}


def load_engsaf(path: Path, columns: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Loads one EngSAF CSV, or every CSV in a folder (train / unseen answers / unseen questions).
    item_id = "<file stem>:<row number>", so IDs stay stable for a given release.
    """
    columns = {**ENGSAF_COLUMNS, **(columns or {})}
    path = Path(path)
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No CSV files in {path}")
    frames = []
    for file in files:
        raw = pd.read_csv(file, dtype=str, keep_default_na=False)
        absent = [c for c in columns.values() if c not in raw.columns]
        if absent:
            raise ValueError(f"{file.name}: columns {absent} not found; it has {list(raw.columns)}. Pass `columns`.")
        frame = pd.DataFrame({key: raw[source].str.strip() for key, source in columns.items()})
        frame["label"] = frame["label"].map(lambda v: ENGSAF_LABELS.get(str(v).strip().split(".")[0]))
        if frame["label"].isna().any():
            raise ValueError(f"{file.name}: labels other than 0/1/2 in {columns['label']}")
        frame.insert(0, "item_id", [f"{file.stem}:{i}" for i in range(len(raw))])
        frames.append(frame)
    return validate(pd.concat(frames, ignore_index=True))


# ---------------------------------------------------------------- Mohler et al. (2011)
# Grades are the average of two graders on 0..5. The cut-offs that turn them into three labels
# are a research decision; they are fixed here, before any results are seen.
MOHLER_CUTOFFS = (2.5, 4.5)  # < 2.5 incorrect, 2.5 .. < 4.5 partial, >= 4.5 correct


def mohler_label(score: float, cutoffs: Tuple[float, float] = MOHLER_CUTOFFS) -> str:
    low, high = cutoffs
    return "incorrect" if score < low else "partial" if score < high else "correct"


def _read_id_lines(file: Path) -> Dict[str, list]:
    """Lines look like "1.1 text"; returns {question_id: [text, ...]} in file order."""
    out: Dict[str, list] = {}
    for line in file.read_text(encoding="latin-1").splitlines():
        if not line.strip():
            continue
        qid, _, text = line.strip().partition(" ")
        out.setdefault(qid, []).append(text.strip())
    return out


def load_mohler(path: Path, cutoffs: Tuple[float, float] = MOHLER_CUTOFFS) -> pd.DataFrame:
    """
    Loads ShortAnswerGrading_v2.0 (path = the folder that contains data/, or data/ itself).
    Uses data/raw/{questions,answers,x.y} and data/scores/x.y/ave; questions commented out
    with '#' in data/docs/files are skipped, as in the authors' own work.
    item_id = "<question id>.<0-based answer index>", matching the dataset's x.y.z notation.
    """
    root = Path(path)
    data = root / "data" if (root / "data").is_dir() else root
    question_ids = [
        line.strip() for line in (data / "docs" / "files").read_text(encoding="latin-1").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    questions = _read_id_lines(data / "raw" / "questions")
    references = _read_id_lines(data / "raw" / "answers")
    rows = []
    for qid in question_ids:
        answers = _read_id_lines(data / "raw" / qid).get(qid, [])
        scores = [float(s) for s in (data / "scores" / qid / "ave").read_text().split()]
        if len(answers) != len(scores):
            raise ValueError(f"Question {qid}: {len(answers)} answers but {len(scores)} scores")
        for i, (answer, score) in enumerate(zip(answers, scores)):
            rows.append({
                "item_id": f"{qid}.{i}",
                "question_id": qid,
                "question": questions[qid][0],
                "reference": references[qid][0],
                "answer": answer,
                "label": mohler_label(score, cutoffs),
            })
    return validate(pd.DataFrame(rows))


# ---------------------------------------------------------------- synthetic (tests and dry runs)
_TOPIC_WORDS = (
    "stack queue heap tree graph hash index cache thread process lock page frame packet "
    "socket router kernel buffer pointer array list node edge vertex key value table row"
).split()


def make_synthetic(n_questions: int = 40, answers_per_question: int = 12, seed: int = 0) -> pd.DataFrame:
    """
    EngSAF-shaped fake data with no real content: correct answers reuse most reference words,
    partial ones about half, incorrect ones few. Labels are skewed towards "correct" like real data.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_questions):
        words = list(rng.choice(_TOPIC_WORDS, size=8, replace=False))
        reference = " ".join(words)
        for a in range(answers_per_question):
            label = rng.choice(LABELS, p=[0.25, 0.3, 0.45])
            keep = {"incorrect": 0.15, "partial": 0.5, "correct": 0.9}[label]
            kept = [w for w in words if rng.random() < keep]
            noise = list(rng.choice(_TOPIC_WORDS, size=3))
            rows.append({
                "item_id": f"syn{q}.{a}",
                "question_id": f"syn{q}",
                "question": f"Synthetic question {q} about {words[0]}?",
                "reference": reference,
                "answer": " ".join(kept + noise),
                "label": str(label),
            })
    return validate(pd.DataFrame(rows))


LOADERS: Dict[str, Callable[..., pd.DataFrame]] = {
    "engsaf": load_engsaf,
    "mohler": load_mohler,
    "synthetic": lambda path=None: make_synthetic(),
}


def load(name: str, path: Optional[Path] = None) -> pd.DataFrame:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(LOADERS)}")
    if name != "synthetic" and path is None:
        raise ValueError(f"--data is required for {name}")
    return LOADERS[name](path) if name != "synthetic" else LOADERS[name]()
