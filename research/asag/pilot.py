"""
Short-answer grading pilot: v1 cosine vs B0 cosine vs rubric LLM grader, on any dataset with a loader.

Stages (run in this order; each one reads what the previous one saved):

    split    stratified sample (by question and label, fixed seed) and the dev/test split.
             Saved once and never overwritten, so the split is fixed before any scoring.
    plan     counts the LLM calls still needed (rubrics + grading). Makes no calls.
    rubrics  one rubric per sampled question, from the question + reference answer only,
             10 questions per call (fits a free-tier daily quota).
    grade    rubric grading: one call per question, all its sampled answers together
             (shuffled with a fixed seed, labelled A1, A2, ..., graded independently).
    check    batching check: 10 fixed dev answers re-graded one per call and compared.
    run      rubrics, then grade, then check, under one --max-calls budget (one day's work);
             prints the calls still needed at the end.
    report   cosine scores (and rubric scores once graded), everything fitted on dev only,
             metrics on test only. Datasets with numeric gold scores (Mohler) get Pearson,
             Spearman and RMSE as primary metrics and a grader-1-vs-grader-2 ceiling.

Examples:
    python -m research.asag.pilot split  --dataset engsaf --data research/data/engsaf
    python -m research.asag.pilot plan   --dataset engsaf --data research/data/engsaf
    python -m research.asag.pilot rubrics --dataset engsaf --data research/data/engsaf --max-calls 120
    python -m research.asag.pilot grade  --dataset engsaf --data research/data/engsaf --max-calls 300
    python -m research.asag.pilot report --dataset engsaf --data research/data/engsaf

    # Whole pipeline on synthetic data with an offline stand-in for Gemini (no API calls):
    python -m research.asag.pilot all --dataset synthetic --offline

Outputs:
    research/results/<dataset>/split.csv          item IDs + split (no text, no labels)
    research/results/<dataset>/rubrics/<qid>.json rubrics (no question or reference text)
    research/results/<dataset>/thresholds.json    dev-fitted cut-offs and linear maps per method
    research/results/<dataset>/test/              metrics.md (the report), metrics.csv, primary_scores.csv, confusion_*.csv
    research/data/<dataset>/grades.csv            per-answer rubric grades (git-ignored)
    research/data/<dataset>/predictions.csv       per-answer scores and labels (git-ignored)
    research/data/llm_cache/                      every Gemini answer, keyed by (model, prompt hash)
Synthetic runs write everything under research/data/synthetic/ instead.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research import evaluate
from research.asag import datasets, rubrics, sampling, thresholds
from research.asag.llm_cache import BudgetExceeded, CachedLLM, QuotaStop

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "research" / "data"
RESULTS_ROOT = ROOT / "research" / "results"
CACHE_DIR = DATA_ROOT / "llm_cache"
OFFLINE_MODEL = "offline-stand-in"

# Label methods compared on test once rubric grades exist. *_tuned use dev-tuned cut-offs; *_app the app's fixed 50/80.
METHODS = ["v1_cosine_tuned", "b0_cosine_tuned", "v1_cosine_app", "b0_cosine_app", "rubric_tuned"]


class Paths:
    def __init__(self, dataset: str):
        synthetic = dataset == "synthetic"
        self.results = (DATA_ROOT / "synthetic" / "results") if synthetic else RESULTS_ROOT / dataset
        self.data = DATA_ROOT / dataset
        self.split = self.results / "split.csv"
        self.rubrics = self.results / "rubrics"
        self.thresholds = self.results / "thresholds.json"
        self.test = self.results / "test"
        self.grades = self.data / "grades.csv"
        self.predictions = self.data / "predictions.csv"


def _safe(question_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", question_id)


# ---------------------------------------------------------------- offline stand-in (no API)
def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _offline_rubric(reference: str) -> Dict:
    words = list(dict.fromkeys(_words(reference)))[:6] or ["answer"]
    return {"key_points": [{"id": f"K{i}", "text": w} for i, w in enumerate(words, 1)],
            "misconceptions": [{"id": "M1", "text": "unrelated claim"}]}


def offline_call(prompt: str, schema=None, system=None, temperature=None, model=None):
    """Deterministic stand-in for Gemini rubric writing, for dry runs on synthetic data only."""
    if prompt.startswith("Write an analytic rubric for each of"):
        blocks = re.findall(r"^Q(\d+)\nQuestion:\n.*?\nReference answer:\n(.*?)(?=\n\nQ\d+\n|\n\nRules for every rubric:)",
                            prompt, flags=re.S | re.M)
        return [{"question": f"Q{n}", **_offline_rubric(ref)} for n, ref in blocks]
    if "Reference answer:" in prompt and "Student answer" not in prompt:
        return _offline_rubric(prompt.split("Reference answer:")[1].split("Rules:")[0])
    raise ValueError("The offline stand-in only writes rubrics; use offline_grade for grading")


def offline_grade(question: str, answer: str, rubric: Dict, llm) -> Dict:
    """Stand-in grader: a key point counts as met when its words appear in the answer."""
    said = set(_words(answer))
    points = [2 if set(_words(k["text"])) <= said else 0 for k in rubric["key_points"]]
    return {"key_point_scores": points, "misconceptions_present": [], "score": 100 * sum(points) / (2 * len(points))}


def offline_grade_batch(question: str, question_id: str, answers: Dict[str, str], rubric: Dict, llm) -> Dict[str, Dict]:
    return {item_id: offline_grade(question, text, rubric, llm) for item_id, text in answers.items()}


# ---------------------------------------------------------------- call grouping (free-tier quota)
# Grading groups: all sampled answers to one question in one call. If True, dev and test answers
# of a question are graded in separate calls instead (about twice the calls).
GROUP_BY_SPLIT = False
# Batching check: this many dev answers are re-graded one per call and compared with their batched grades
CHECK_N = 10
CHECK_SEED = 42


def grading_groups(sample: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    """[(group key, rows)] in a fixed order; the key is the question id (plus the split if GROUP_BY_SPLIT)."""
    keys = ["question_id", "split"] if GROUP_BY_SPLIT else ["question_id"]
    groups = []
    for key, rows in sample.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        groups.append(("/".join(key), rows.sort_values("item_id")))
    return groups


def _rubric_path(paths: Paths, question_id: str) -> Path:
    return paths.rubrics / f"{_safe(question_id)}.json"


def _batch_inputs(questions: pd.DataFrame, batch: List[str]) -> List[Dict[str, str]]:
    by_id = questions.set_index("question_id")
    return [{"question": by_id.loc[q, "question"], "reference": by_id.loc[q, "reference"]} for q in batch]


def check_ids(sample: pd.DataFrame, paths: Paths) -> List[str]:
    """The CHECK_N dev answers for the batching check, drawn once with a fixed seed and saved."""
    file = paths.results / "batch_check_ids.csv"
    if file.exists():
        return pd.read_csv(file, dtype=str)["item_id"].tolist()
    dev = sorted(sample.loc[sample["split"] == "dev", "item_id"])
    chosen = sorted(np.random.default_rng(CHECK_SEED).choice(dev, size=min(CHECK_N, len(dev)), replace=False).tolist())
    file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"item_id": chosen}).to_csv(file, index=False)
    return chosen


# ---------------------------------------------------------------- stages
def stage_split(df: pd.DataFrame, paths: Paths, force: bool = False) -> pd.DataFrame:
    if paths.split.exists() and not force:
        print(f"Split already fixed at {paths.split}; not changing it.")
        return sampling.load_split(paths.split)
    split = sampling.make_split(df)
    sampling.save_split(split, paths.split)
    counts = split["split"].value_counts().to_dict()
    print(f"Saved split ({counts}) to {paths.split}")
    return split


def _sample(df: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    if not paths.split.exists():
        raise SystemExit("No split yet: run the `split` stage first.")
    return sampling.apply_split(df, sampling.load_split(paths.split))


def _questions(sample: pd.DataFrame) -> pd.DataFrame:
    return sample.drop_duplicates("question_id")[["question_id", "question", "reference"]].sort_values("question_id")


def _graded(paths: Paths) -> pd.DataFrame:
    return pd.read_csv(paths.grades, dtype={"item_id": str}) if paths.grades.exists() else pd.DataFrame(columns=["item_id"])


def stage_plan(df: pd.DataFrame, paths: Paths, llm: CachedLLM, offline: bool) -> Dict[str, int]:
    """Counts the API calls the remaining stages need. Makes no calls."""
    from research.asag import grader

    sample = _sample(df, paths)
    questions = _questions(sample)
    rubric_calls = 0
    for batch in rubrics.batches(questions["question_id"].tolist()):
        if all(_rubric_path(paths, q).exists() for q in batch):
            continue
        prompt = rubrics.batch_prompt(_batch_inputs(questions, batch))
        rubric_calls += not llm.is_cached(prompt, rubrics.BATCH_SCHEMA, rubrics.GENERATION_SYSTEM)
    done = set(_graded(paths)["item_id"])
    grade_calls = 0
    for _, rows in grading_groups(sample):
        if offline or set(rows["item_id"]) <= done:
            continue
        qid = rows["question_id"].iloc[0]
        if not _rubric_path(paths, qid).exists():
            grade_calls += 1  # needs the rubric first, so it cannot be cached yet
            continue
        answers = dict(zip(rows["item_id"], rows["answer"]))
        grade_calls += not grader.is_batch_cached(rows["question"].iloc[0], qid, answers, rubrics.load_rubric(_rubric_path(paths, qid)), llm)
    check_calls = 0
    if not offline:
        by_id = sample.set_index("item_id")
        for item_id in check_ids(sample, paths):
            row = by_id.loc[item_id]
            rubric_file = _rubric_path(paths, row["question_id"])
            if not (rubric_file.exists() and grader.is_cached(row["question"], row["answer"], rubrics.load_rubric(rubric_file), llm)):
                check_calls += 1
    plan = {"questions": len(questions), "answers": len(sample), "rubric_calls": rubric_calls,
            "grading_calls": grade_calls, "check_calls": check_calls,
            "total_calls": rubric_calls + grade_calls + check_calls}
    print(f"Model: {llm.model}")
    print(f"Sample: {plan['answers']} answers ({(sample['split'] == 'dev').sum()} dev / {(sample['split'] == 'test').sum()} test) "
          f"to {plan['questions']} questions")
    print(f"LLM calls still needed: {rubric_calls} rubric batches + {grade_calls} grading groups + {check_calls} batching-check "
          f"= {plan['total_calls']}")
    return plan


def stage_rubrics(df: pd.DataFrame, paths: Paths, llm: CachedLLM, only: Optional[List[str]] = None) -> int:
    """One call per batch of up to rubrics.BATCH_SIZE questions (question + reference answer only)."""
    questions = _questions(_sample(df, paths))
    made = 0
    for index, batch in enumerate(rubrics.batches(questions["question_id"].tolist())):
        if only is not None and not set(batch) & set(only):
            continue
        if all(_rubric_path(paths, q).exists() for q in batch):
            continue
        inputs = _batch_inputs(questions, batch)
        generated = rubrics.generate_rubric_batch(inputs, llm)
        key = llm.key(rubrics.batch_prompt(inputs), rubrics.BATCH_SCHEMA, rubrics.GENERATION_SYSTEM)
        for qid, rubric in zip(batch, generated):
            rubrics.save_rubric(_rubric_path(paths, qid), qid, {**rubric, "batch": index}, llm.model, key)
            made += 1
    print(f"Rubrics: {made} new, in {paths.rubrics}")
    return made


def _grade_row(item_id: str, result: Dict) -> Dict:
    return {"item_id": item_id, "rubric_score": result["score"],
            "key_point_scores": json.dumps(result["key_point_scores"]),
            "key_point_scores_raw": json.dumps(result.get("key_point_scores_raw", result["key_point_scores"])),
            "key_points_rejected": json.dumps(result.get("key_points_rejected", [])),
            "misconceptions_present": json.dumps(result["misconceptions_present"]),
            "misconceptions_rejected": json.dumps(result.get("misconceptions_rejected", [])),
            "evidence": json.dumps(result.get("evidence", {}), ensure_ascii=False),
            "batch_label": result.get("batch_label", ""), "batch_size": result.get("batch_size", "")}


def stage_grade(df: pd.DataFrame, paths: Paths, llm: CachedLLM, offline: bool,
                only: Optional[List[str]] = None) -> pd.DataFrame:
    """One call per grading group (see grading_groups). only: restrict to these question ids."""
    sample = _sample(df, paths)
    if offline:
        grade_batch = offline_grade_batch
    else:
        from research.asag import grader
        grade_batch = grader.grade_batch
    rows = _graded(paths).to_dict("records")
    seen = {r["item_id"] for r in rows}
    paths.grades.parent.mkdir(parents=True, exist_ok=True)
    try:
        for _, group in grading_groups(sample):
            qid = group["question_id"].iloc[0]
            if (only is not None and qid not in only) or set(group["item_id"]) <= seen:
                continue
            if not _rubric_path(paths, qid).exists():
                raise SystemExit(f"No rubric for question {qid}: run the `rubrics` stage first.")
            answers = dict(zip(group["item_id"], group["answer"]))
            results = grade_batch(group["question"].iloc[0], qid, answers, rubrics.load_rubric(_rubric_path(paths, qid)), llm)
            for item_id in group["item_id"]:
                if item_id not in seen:
                    rows.append(_grade_row(item_id, results[item_id]))
                    seen.add(item_id)
    finally:
        # Saved even when the run stops on quota or budget, so the next run resumes
        pd.DataFrame(rows).to_csv(paths.grades, index=False)
    print(f"Grades: {len(rows)} of {len(sample)} answers graded, in {paths.grades}")
    return pd.DataFrame(rows)


def stage_check(df: pd.DataFrame, paths: Paths, llm: CachedLLM, only: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Batching check: re-grades the fixed CHECK_N dev answers one per call and compares each with its
    batched grade. Writes research/results/<dataset>/batch_check.md (scores only, no text).
    """
    from research.asag import grader

    sample = _sample(df, paths).set_index("item_id")
    batched = _graded(paths).set_index("item_id")
    check_file = paths.data / "check_grades.csv"
    rows = pd.read_csv(check_file, dtype={"item_id": str}).to_dict("records") if check_file.exists() else []
    done = {r["item_id"] for r in rows}
    waiting = []
    try:
        for item_id in check_ids(sample.reset_index(), paths):
            row = sample.loc[item_id]
            if only is not None and row["question_id"] not in only:
                continue
            if item_id not in batched.index:
                waiting.append(item_id)
                continue
            if item_id in done:
                continue
            rubric = rubrics.load_rubric(_rubric_path(paths, row["question_id"]))
            result = grader.grade_answer(row["question"], row["answer"], rubric, llm)
            rows.append(_grade_row(item_id, result))
    finally:
        if rows:
            check_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(check_file, index=False)
    if waiting:
        print(f"Batching check: {len(waiting)} of {CHECK_N} answers have no batched grade yet.")
    if not rows:
        return None
    single = pd.DataFrame(rows).set_index("item_id")
    compared = []
    for item_id in single.index:
        b, s = batched.loc[item_id], single.loc[item_id]
        kb, ks = json.loads(b["key_point_scores"]), json.loads(s["key_point_scores"])
        compared.append({"item_id": item_id, "batched_score": float(b["rubric_score"]), "single_score": float(s["rubric_score"]),
                         "difference": float(b["rubric_score"]) - float(s["rubric_score"]),
                         "key_point_marks_equal": sum(x == y for x, y in zip(kb, ks)), "key_points": len(kb)})
    table = pd.DataFrame(compared)
    diff = table["difference"]
    marks_equal = table["key_point_marks_equal"].sum() / table["key_points"].sum()
    lines = [
        f"# Batching check ({len(table)} of {CHECK_N} dev answers)", "",
        "Each answer was graded once inside its question's batch and once on its own (same rubric, temperature 0).", "",
        f"- Mean absolute score difference: {diff.abs().mean():.1f} points (0-100 scale); mean signed (batched - single): {diff.mean():+.1f}",
        f"- Identical scores: {(diff == 0).sum()} of {len(table)}; largest difference: {diff.abs().max():.1f}",
        f"- Key-point marks identical: {marks_equal:.0%}", "",
        "| Answer | Batched | Single | Difference | Key-point marks equal |", "|---|---|---|---|---|",
    ] + [f"| {r.item_id} | {r.batched_score:.1f} | {r.single_score:.1f} | {r.difference:+.1f} | {r.key_point_marks_equal}/{r.key_points} |"
         for r in table.itertuples()]
    (paths.results / "batch_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:7]))
    return table


def cosine_scores(sample: pd.DataFrame, encode: Optional[Callable] = None) -> np.ndarray:
    """Cosine between each answer and its reference answer, with the app's Sentence-BERT model."""
    if encode is None:
        from backend.app.services.semantic import get_sbert_model
        model = get_sbert_model()
        encode = lambda texts: model.encode(texts, batch_size=32, show_progress_bar=False)
    a = np.asarray(encode(sample["answer"].tolist()), dtype=float)
    r = np.asarray(encode(sample["reference"].tolist()), dtype=float)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(r, axis=1)
    return np.where(norms > 0, (a * r).sum(axis=1) / np.where(norms > 0, norms, 1), 0.0)


def linear_map(dev_scores: Sequence[float], dev_gold: Sequence[float], low: float, high: float) -> Dict[str, float]:
    """Least-squares line gold ~ slope * score + intercept, fitted on dev only; predictions are clipped to [low, high]."""
    x = np.asarray(dev_scores, dtype=float)
    if np.ptp(x) == 0:
        return {"slope": 0.0, "intercept": float(np.mean(dev_gold)), "low": low, "high": high}
    slope, intercept = np.polyfit(x, np.asarray(dev_gold, dtype=float), 1)
    return {"slope": float(slope), "intercept": float(intercept), "low": low, "high": high}


def apply_map(scores: Sequence[float], m: Dict[str, float]) -> np.ndarray:
    return np.clip(np.asarray(scores, dtype=float) * m["slope"] + m["intercept"], m["low"], m["high"])


def _available_methods(sample: pd.DataFrame, paths: Paths) -> List[str]:
    """Cosine methods always; the rubric grader only once every sampled answer is graded."""
    bases = ["v1", "b0"]
    if paths.grades.exists():
        grades = pd.read_csv(paths.grades, dtype={"item_id": str})
        graded = set(grades["item_id"])
        if set(sample["item_id"]) <= graded:
            bases.append("rubric")
        else:
            print(f"Rubric grader left out: {len(set(sample['item_id']) - graded)} sampled answers not graded yet.")
    else:
        print("Rubric grader left out: no grades yet (run `grade`).")
    return bases


def stage_report(df: pd.DataFrame, paths: Paths, dataset: str, encode: Optional[Callable] = None) -> pd.DataFrame:
    """
    Writes the test-split report. Everything fitted (label cut-offs, the linear map to the
    gold scale) uses dev only. Datasets with numeric gold scores get the continuous metrics
    as primary; the three labels are then secondary. Returns the label-metrics table.
    """
    from backend.app.services.semantic import similarity_from_cosine, similarity_from_cosine_b0

    sample = _sample(df, paths)
    bases = _available_methods(sample, paths)
    if "rubric" in bases:
        grades = pd.read_csv(paths.grades, dtype={"item_id": str})
        sample = sample.merge(grades[["item_id", "rubric_score"]], on="item_id", how="left")

    cos = cosine_scores(sample, encode)
    sample["cosine"] = cos
    sample["v1_score"] = [similarity_from_cosine(c) for c in cos]
    sample["b0_score"] = [similarity_from_cosine_b0(c) for c in cos]

    def tuned(base: str) -> str:
        return "rubric_tuned" if base == "rubric" else f"{base}_cosine_tuned"

    methods = [tuned(b) for b in bases if b != "rubric"] + [f"{b}_cosine_app" for b in ("v1", "b0")]
    methods += [tuned("rubric")] if "rubric" in bases else []

    dev = sample[sample["split"] == "dev"]
    cutoffs = {b: thresholds.tune(dev[f"{b}_score"], dev["label"]) for b in bases}
    for b, c in cutoffs.items():
        sample[tuned(b)] = thresholds.labels_for(sample[f"{b}_score"], (c["low"], c["high"]))
    for b in ("v1", "b0"):
        sample[f"{b}_cosine_app"] = thresholds.labels_for(sample[f"{b}_score"], thresholds.APP_CUTOFFS)
    for method in methods:
        sample[f"{method}_score"] = sample[f"{method.split('_')[0]}_score"]

    continuous = "score" in sample.columns
    maps = {}
    if continuous:
        if dataset not in datasets.SCORE_SCALES:
            raise SystemExit(f"Add the gold score range for {dataset} to datasets.SCORE_SCALES")
        low, high = datasets.SCORE_SCALES[dataset]
        for b in bases:
            maps[b] = linear_map(dev[f"{b}_score"], dev["score"], low, high)
            sample[f"{b}_mapped"] = apply_map(sample[f"{b}_score"], maps[b])

    paths.thresholds.parent.mkdir(parents=True, exist_ok=True)
    paths.thresholds.write_text(json.dumps({
        "fitted_on": "dev",
        "label_cutoffs": {"objective": "QWK (ties: accuracy)", "app_cutoffs": thresholds.APP_CUTOFFS, **cutoffs},
        "linear_maps_to_gold_scale": maps or None,
    }, indent=2), encoding="utf-8")
    paths.predictions.parent.mkdir(parents=True, exist_ok=True)
    sample.drop(columns=["question", "reference", "answer"]).to_csv(paths.predictions, index=False)

    test = sample[sample["split"] == "test"].rename(columns={"label": "human"})
    has_graders = {"score_grader_1", "score_grader_2"} <= set(test.columns)
    paths.test.mkdir(parents=True, exist_ok=True)
    report = []

    if continuous:
        rows = [{"method": b, **evaluate.evaluate_continuous(test["score"], test[f"{b}_mapped"])} for b in bases]
        if has_graders:
            rows.append({"method": "human ceiling: grader 1 vs grader 2",
                         **evaluate.evaluate_continuous(test["score_grader_1"], test["score_grader_2"])})
        primary = pd.DataFrame(rows)
        primary.to_csv(paths.test / "primary_scores.csv", index=False)
        scale = f"{low:g}-{high:g}"
        report.append(evaluate.continuous_markdown(
            primary, f"{dataset}: primary — scores vs gold ({scale}) on the test split (n={len(test)})", scale))
        report.append(
            "Each method's raw score is mapped to the gold scale by a straight line fitted on dev only "
            "(clipped to the scale). Pearson and Spearman do not depend on that map; RMSE does. "
            "The ceiling row compares the two human graders with each other, not with their average.\n")

    labels = list(datasets.LABELS)
    table = evaluate.evaluate_frame(test, methods, labels)
    if has_graders:
        g1 = [datasets.mohler_label(s) for s in test["score_grader_1"]]
        g2 = [datasets.mohler_label(s) for s in test["score_grader_2"]]
        ceiling = evaluate.evaluate_method(g1, g2, labels, scores=test["score_grader_2"].tolist())
        table = pd.concat([table, pd.DataFrame([{"method": "human ceiling: grader 1 vs grader 2",
                                                 "spearman_on": "score_grader_2", **ceiling}])], ignore_index=True)
    table.to_csv(paths.test / "metrics.csv", index=False)
    kind = "secondary — three labels" if continuous else "agreement with human labels"
    title = f"{dataset}: {kind} on the test split (n={len(test)})"
    report.append(evaluate.metrics_markdown(table, title))
    report.append(
        "*_tuned: label cut-offs chosen on dev only (max QWK). *_app: the app's fixed cut-offs (50 / 80). "
        "With tuned cut-offs v1 and B0 give the same labels, because both are increasing functions of the cosine.\n")
    y_true = evaluate.to_ordinal(test["human"].tolist(), labels)
    for method in methods:
        matrix = evaluate.confusion(y_true, evaluate.to_ordinal(test[method].tolist(), labels), labels)
        matrix.to_csv(paths.test / f"confusion_{method}.csv")
        report.append(evaluate.confusion_markdown(matrix, method))
    (paths.test / "metrics.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[:3] if continuous else report[:1]))
    print(f"Report: {paths.test / 'metrics.md'}\nFitted on dev only: {paths.thresholds}")
    return table


# ---------------------------------------------------------------- CLI
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=["split", "plan", "rubrics", "grade", "check", "run", "report", "all"])
    parser.add_argument("--dataset", required=True, choices=sorted(datasets.LOADERS))
    parser.add_argument("--data", type=Path, help="dataset file or folder (in research/data/)")
    parser.add_argument("--max-calls", type=int, default=0, help="approved number of new LLM calls (default 0: cache only)")
    parser.add_argument("--min-interval", type=float, default=4.0, help="seconds between API calls (rate limit)")
    parser.add_argument("--model", help="Gemini model (default: GEMINI_MODEL)")
    parser.add_argument("--offline", action="store_true", help="offline stand-in for Gemini; synthetic data only")
    parser.add_argument("--force-split", action="store_true", help="re-draw the split (only before any scoring)")
    args = parser.parse_args(argv)

    if args.offline and args.dataset != "synthetic":
        parser.error("--offline is only for the synthetic dataset")
    if args.stage == "all" and not args.offline:
        parser.error("`all` is only for offline dry runs; run the stages one by one so calls can be approved")

    df = datasets.load(args.dataset, args.data)
    paths = Paths(args.dataset)
    if args.offline:
        llm = CachedLLM(paths.data / "offline_cache", model=OFFLINE_MODEL, max_calls=10**6, call=offline_call)
    else:
        llm = CachedLLM(CACHE_DIR, model=args.model, max_calls=args.max_calls, min_interval_s=args.min_interval)

    try:
        if args.stage in ("split", "all"):
            stage_split(df, paths, args.force_split)
        if args.stage in ("plan", "all"):
            stage_plan(df, paths, llm, args.offline)
        if args.stage in ("rubrics", "all"):
            stage_rubrics(df, paths, llm)
        if args.stage == "run":
            # One day's work under one budget: rubrics, then grading, then the batching check
            stage_rubrics(df, paths, llm)
            stage_grade(df, paths, llm, args.offline)
            stage_check(df, paths, llm)
        if args.stage in ("grade", "all"):
            stage_grade(df, paths, llm, args.offline)
        if args.stage == "check":
            stage_check(df, paths, llm)
        if args.stage in ("report", "all"):
            stage_report(df, paths, args.dataset)
    except (BudgetExceeded, QuotaStop) as e:
        print(f"Stopped: {e}\nEverything so far is cached; re-run the same stage to resume.", file=sys.stderr)
        return 2
    finally:
        if llm.calls or llm.hits:
            print(f"LLM: {llm.calls} new calls, {llm.hits} from cache, {llm.retries} retries after refusals")
            tokens = llm.tokens()
            print("Tokens (new calls): " + ", ".join(f"{k.replace('_token_count', '')} {v}" for k, v in tokens.items()))
        if args.stage == "run":
            stage_plan(df, paths, llm, args.offline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
