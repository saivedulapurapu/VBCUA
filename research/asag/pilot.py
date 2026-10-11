"""
Short-answer grading pilot: v1 cosine vs B0 cosine vs rubric LLM grader, on any dataset with a loader.

Stages (run in this order; each one reads what the previous one saved):

    split    stratified sample (by question and label, fixed seed) and the dev/test split.
             Saved once and never overwritten, so the split is fixed before any scoring.
    plan     counts the LLM calls still needed (rubrics + grading). Makes no calls.
    rubrics  one rubric per sampled question, from the question + reference answer only.
    grade    rubric grading of every sampled answer.
    report   cosine scores, cut-offs tuned on dev only, metrics on test only.

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
    research/results/<dataset>/thresholds.json    dev-tuned cut-offs per method
    research/results/<dataset>/test/              metrics.csv, metrics.md, confusion_*.csv
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
from typing import Callable, Dict, List, Optional

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

# Methods compared on test. *_tuned use dev-tuned cut-offs; *_app use the app's fixed 50/80.
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


def offline_call(prompt: str, schema=None, system=None, temperature=None, model=None) -> Dict:
    """Deterministic stand-in for Gemini, for dry runs on synthetic data only."""
    if "Reference answer:" in prompt and "Student answer:" not in prompt:
        reference = prompt.split("Reference answer:")[1].split("Rules:")[0]
        words = list(dict.fromkeys(_words(reference)))[:6] or ["answer"]
        return {"key_points": [{"id": f"K{i}", "text": w} for i, w in enumerate(words, 1)],
                "misconceptions": [{"id": "M1", "text": "unrelated claim"}]}
    raise ValueError("The offline stand-in only writes rubrics; use offline_grade for grading")


def offline_grade(question: str, answer: str, rubric: Dict, llm) -> Dict:
    """Stand-in grader: a key point counts as met when its words appear in the answer."""
    said = set(_words(answer))
    points = [2 if set(_words(k["text"])) <= said else 0 for k in rubric["key_points"]]
    return {"key_point_scores": points, "misconceptions_present": [], "score": 100 * sum(points) / (2 * len(points))}


def _grader(offline: bool) -> Callable:
    if offline:
        return offline_grade
    from research.asag import grader  # the rubric grader (added in its own commit)
    return grader.grade_answer


def _grade_prompt_cached(offline: bool) -> Optional[Callable]:
    if offline:
        return None
    from research.asag import grader
    return grader.is_cached


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


def stage_plan(df: pd.DataFrame, paths: Paths, llm: CachedLLM, offline: bool) -> Dict[str, int]:
    """Counts the API calls the remaining stages need. Makes no calls."""
    sample = _sample(df, paths)
    questions = _questions(sample)
    rubric_calls = sum(
        not (paths.rubrics / f"{_safe(q.question_id)}.json").exists()
        and not llm.is_cached(rubrics.generation_prompt(q.question, q.reference), rubrics.GENERATION_SCHEMA, rubrics.GENERATION_SYSTEM)
        for q in questions.itertuples()
    )
    done = set(pd.read_csv(paths.grades, dtype=str)["item_id"]) if paths.grades.exists() else set()
    grade_calls = 0
    is_cached = None
    for row in sample.itertuples():
        if offline or row.item_id in done:
            continue
        rubric_file = paths.rubrics / f"{_safe(row.question_id)}.json"
        if not rubric_file.exists():
            grade_calls += 1  # the grading prompt needs the rubric, so it cannot be cached yet
            continue
        is_cached = is_cached or _grade_prompt_cached(offline)
        if not is_cached(row.question, row.answer, rubrics.load_rubric(rubric_file), llm):
            grade_calls += 1
    plan = {"questions": len(questions), "answers": len(sample), "rubric_calls": rubric_calls,
            "grading_calls": grade_calls, "total_calls": rubric_calls + grade_calls}
    print(f"Model: {llm.model}")
    print(f"Sample: {plan['answers']} answers ({(sample['split'] == 'dev').sum()} dev / {(sample['split'] == 'test').sum()} test) "
          f"to {plan['questions']} questions")
    print(f"LLM calls still needed: {plan['rubric_calls']} rubric + {plan['grading_calls']} grading = {plan['total_calls']}")
    return plan


def stage_rubrics(df: pd.DataFrame, paths: Paths, llm: CachedLLM) -> int:
    questions = _questions(_sample(df, paths))
    made = 0
    for q in questions.itertuples():
        path = paths.rubrics / f"{_safe(q.question_id)}.json"
        if path.exists():
            continue
        # Only the question and the reference answer go into the prompt
        rubric = rubrics.generate_rubric(q.question, q.reference, llm)
        key = llm.key(rubrics.generation_prompt(q.question, q.reference), rubrics.GENERATION_SCHEMA, rubrics.GENERATION_SYSTEM)
        rubrics.save_rubric(path, q.question_id, rubric, llm.model, key)
        made += 1
    print(f"Rubrics: {made} new, {len(questions) - made} already saved, in {paths.rubrics}")
    return made


def stage_grade(df: pd.DataFrame, paths: Paths, llm: CachedLLM, offline: bool) -> pd.DataFrame:
    sample = _sample(df, paths)
    grade = _grader(offline)
    done = pd.read_csv(paths.grades, dtype={"item_id": str}) if paths.grades.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    seen = {r["item_id"] for r in rows}
    paths.grades.parent.mkdir(parents=True, exist_ok=True)
    try:
        for row in sample.itertuples():
            if row.item_id in seen:
                continue
            rubric_file = paths.rubrics / f"{_safe(row.question_id)}.json"
            if not rubric_file.exists():
                raise SystemExit(f"No rubric for question {row.question_id}: run the `rubrics` stage first.")
            result = grade(row.question, row.answer, rubrics.load_rubric(rubric_file), llm)
            rows.append({"item_id": row.item_id, "rubric_score": result["score"],
                         "key_point_scores": json.dumps(result["key_point_scores"]),
                         "misconceptions_present": json.dumps(result["misconceptions_present"])})
    finally:
        # Saved even when the run stops on quota or budget, so the next run resumes
        pd.DataFrame(rows).to_csv(paths.grades, index=False)
    print(f"Grades: {len(rows)} of {len(sample)} answers graded, in {paths.grades}")
    return pd.DataFrame(rows)


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


def stage_report(df: pd.DataFrame, paths: Paths, dataset: str, encode: Optional[Callable] = None) -> pd.DataFrame:
    from backend.app.services.semantic import similarity_from_cosine, similarity_from_cosine_b0

    sample = _sample(df, paths)
    if not paths.grades.exists():
        raise SystemExit("No grades yet: run the `grade` stage first.")
    grades = pd.read_csv(paths.grades, dtype={"item_id": str})
    sample = sample.merge(grades[["item_id", "rubric_score"]], on="item_id", how="left")
    if sample["rubric_score"].isna().any():
        raise SystemExit(f"{int(sample['rubric_score'].isna().sum())} sampled answers are not graded yet.")

    cos = cosine_scores(sample, encode)
    sample["cosine"] = cos
    sample["v1_score"] = [similarity_from_cosine(c) for c in cos]
    sample["b0_score"] = [similarity_from_cosine_b0(c) for c in cos]

    dev = sample[sample["split"] == "dev"]
    cutoffs = {name: thresholds.tune(dev[f"{name}_score"], dev["label"]) for name in ("v1", "b0", "rubric")}
    for name, c in cutoffs.items():
        sample[f"{name}_cosine_tuned" if name != "rubric" else "rubric_tuned"] = thresholds.labels_for(sample[f"{name}_score"], (c["low"], c["high"]))
    for name in ("v1", "b0"):
        sample[f"{name}_cosine_app"] = thresholds.labels_for(sample[f"{name}_score"], thresholds.APP_CUTOFFS)
    for method in METHODS:
        base = method.split("_")[0]
        sample[f"{method}_score"] = sample[f"{base}_score"]

    paths.thresholds.parent.mkdir(parents=True, exist_ok=True)
    paths.thresholds.write_text(json.dumps(
        {"tuned_on": "dev", "objective": "QWK (ties: accuracy)", "app_cutoffs": thresholds.APP_CUTOFFS, **cutoffs}, indent=2),
        encoding="utf-8")
    paths.predictions.parent.mkdir(parents=True, exist_ok=True)
    sample.drop(columns=["question", "reference", "answer"]).to_csv(paths.predictions, index=False)

    test = sample[sample["split"] == "test"].rename(columns={"label": "human"})
    title = f"{dataset}: agreement with human labels on the test split (n={len(test)})"
    table = evaluate.write_report(test, METHODS, list(datasets.LABELS), paths.test, title)
    print(evaluate.metrics_markdown(table, title))
    print(f"Cut-offs (dev only): {paths.thresholds}")
    return table


# ---------------------------------------------------------------- CLI
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=["split", "plan", "rubrics", "grade", "report", "all"])
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
        if args.stage in ("grade", "all"):
            stage_grade(df, paths, llm, args.offline)
        if args.stage in ("report", "all"):
            stage_report(df, paths, args.dataset)
    except (BudgetExceeded, QuotaStop) as e:
        print(f"Stopped: {e}\nEverything so far is cached; re-run the same stage to resume.", file=sys.stderr)
        return 2
    finally:
        if llm.calls or llm.hits:
            print(f"LLM: {llm.calls} new calls, {llm.hits} from cache, {llm.retries} retries after refusals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
