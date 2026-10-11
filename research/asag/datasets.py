"""
Short-answer grading datasets in one common shape, so the pilot runs on any of them.

Every loader returns a DataFrame with these columns (one row per student answer):
    item_id      unique, stable ID of the answer
    question_id  ID of the question
    question     question text
    reference    reference (instructor) answer
    answer       student answer
    label        "incorrect" / "partial" / "correct"
and, for datasets with numeric grades (Mohler), optional float columns:
    score            the gold score (Mohler: score_avg, 0-5)
    score_grader_1   grader 1 on the same scale
    score_grader_2   grader 2 on the same scale

Data files stay in research/data/ (git-ignored) and are never committed.
"""
import re
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LABELS = ("incorrect", "partial", "correct")
COLUMNS = ["item_id", "question_id", "question", "reference", "answer", "label"]
SCORE_COLUMNS = ["score", "score_grader_1", "score_grader_2"]
# Fixed range of the gold score per dataset (never taken from the data, so test rows cannot leak in)
SCORE_SCALES = {"mohler": (0.0, 5.0), "mohler_raw": (0.0, 5.0), "synthetic": (0.0, 5.0)}


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Checks the common shape; returns the common columns as strings plus any score columns as floats."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns {missing}")
    scores = [c for c in SCORE_COLUMNS if c in df.columns]
    df = df[COLUMNS + scores].copy()
    for column in COLUMNS:
        df[column] = df[column].astype(str)
    for column in scores:
        df[column] = df[column].astype(float)
        if df[column].isna().any():
            raise ValueError(f"Missing values in {column}")
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
# Grades are the average of two graders on 0..5. The primary metrics use that number directly;
# the three labels (a secondary metric) use cut-offs fixed before any results were seen.
MOHLER_CUTOFFS = (2.5, 4.5)  # < 2.5 incorrect, 2.5 .. < 4.5 partial, >= 4.5 correct
# Exams E11 and E12 were graded 0..10 by each grader; the averages are already on 0..5
MOHLER_TEN_POINT = ("E11", "E12")


def mohler_label(score: float, cutoffs: Tuple[float, float] = MOHLER_CUTOFFS) -> str:
    low, high = cutoffs
    return "incorrect" if score < low else "partial" if score < high else "correct"


# Penn Treebank bracket tokens left by the authors' tokenizer in the cleaned subset
PTB_BRACKETS = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}"}


def _clean_mohler_text(text: str) -> str:
    """
    Undoes the cleaned subset's markup: <STOP> sentence markers become spaces and bracket tokens
    (-LRB- etc.) become the characters they stand for, without the tokenizer's inner spaces
    ("f -LRB- x -RRB-" -> "f (x)"). The words themselves are kept as released.
    """
    text = re.sub(r"<STOP>", " ", str(text))
    for token, char in PTB_BRACKETS.items():
        text = text.replace(token, char)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def load_mohler(path: Path, cutoffs: Tuple[float, float] = MOHLER_CUTOFFS) -> pd.DataFrame:
    """
    Loads the Hugging Face release nkazi/MohlerASAG, "cleaned" subset, open_ended split
    (path = the parquet file, or a folder holding cleaned-oe-*.parquet). Closed-ended questions
    are excluded, as in the authors' own work.
    item_id = the release's id (EXX.QXX.AXX); question_id = EXX.QXX.
    score = score_avg (0..5). Grader scores for E11/E12 are halved to 0..5. Note that for those
    exams the authors' score_avg is not exactly the mean of the halved grades (it is 0.125-0.5
    higher in about 200 rows), so the human ceiling compares grader 1 with grader 2 directly.
    """
    path = Path(path)
    files = sorted(path.glob("cleaned-oe-*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No cleaned-oe-*.parquet in {path}")
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ten_point = raw["id"].str[:3].isin(MOHLER_TEN_POINT)
    scale = np.where(ten_point, 2.0, 1.0)
    df = pd.DataFrame({
        "item_id": raw["id"],
        "question_id": raw["id"].str[:7],
        "question": raw["question"].map(_clean_mohler_text),
        "reference": raw["instructor_answer"].map(_clean_mohler_text),
        "answer": raw["student_answer"].map(_clean_mohler_text),
        "label": [mohler_label(s, cutoffs) for s in raw["score_avg"]],
        "score": raw["score_avg"].astype(float),
        "score_grader_1": raw["score_grader_1"].astype(float) / scale,
        "score_grader_2": raw["score_grader_2"].astype(float) / scale,
    })
    return validate(df)


def _read_id_lines(file: Path) -> Dict[str, list]:
    """Lines look like "1.1 text"; returns {question_id: [text, ...]} in file order."""
    out: Dict[str, list] = {}
    for line in file.read_text(encoding="latin-1").splitlines():
        if not line.strip():
            continue
        qid, _, text = line.strip().partition(" ")
        out.setdefault(qid, []).append(text.strip())
    return out


def load_mohler_raw(path: Path, cutoffs: Tuple[float, float] = MOHLER_CUTOFFS) -> pd.DataFrame:
    """
    Loads the original ShortAnswerGrading_v2.0 archive (path = the folder that contains data/,
    or data/ itself), as an alternative to the Hugging Face release. Uses data/raw and
    data/scores/x.y/ave; questions commented out with '#' in data/docs/files are skipped.
    item_id = "<question id>.<0-based answer index>", the dataset's x.y.z notation.
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
                "score": score,
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
    "mohler_raw": load_mohler_raw,
    "synthetic": lambda path=None: make_synthetic(),
}


def load(name: str, path: Optional[Path] = None) -> pd.DataFrame:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(LOADERS)}")
    if name != "synthetic" and path is None:
        raise ValueError(f"--data is required for {name}")
    return LOADERS[name](path) if name != "synthetic" else LOADERS[name]()
