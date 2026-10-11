"""
Smoke test of the batched design on the first rubric batch, before the full run.

Runs exactly what the pilot stages run, restricted to the questions of rubric batch 0 (same
prompts, cache, rubric files and grades.csv, so nothing is paid twice):
  1 rubric call for the batch, one grading call per question (--questions, default all of the
  batch), and the batching check for those of the 10 fixed check answers that belong to them.
The report shows the rubrics and the gradings of DEV answers only; grades of test answers that
share a grading call are stored for the final report but not shown or analysed here.

    python -m research.asag.smoke --dataset mohler --data research/data/mohler --max-calls 20

The report (with answer text) goes to research/data/<dataset>/smoke.md, which is git-ignored.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from research.asag import datasets, grader, pilot, rubrics
from research.asag.llm_cache import BudgetExceeded, CachedLLM, QuotaStop


def evidence_in_answer(quote: str, answer: str) -> Optional[bool]:
    """The grader's verbatim check; None for an empty quote (nothing claimed)."""
    return None if not (quote or "").strip() else grader.quote_in_answer(quote, answer)


def smoke_questions(sample: pd.DataFrame, n_questions: Optional[int] = None) -> List[str]:
    first = rubrics.batches(sorted(sample["question_id"].unique()))[0]
    return first[:n_questions] if n_questions else first


def run(df: pd.DataFrame, paths: "pilot.Paths", llm: CachedLLM, n_questions: Optional[int] = None) -> Dict:
    sample = pilot._sample(df, paths)
    questions = smoke_questions(sample, n_questions)
    out = {"questions": questions, "stopped": None, "failures": []}
    try:
        pilot.stage_rubrics(df, paths, llm, only=questions)
        pilot.stage_grade(df, paths, llm, offline=False, only=questions)
        pilot.stage_check(df, paths, llm, only=questions)
    except (BudgetExceeded, QuotaStop) as e:
        out["stopped"] = str(e)  # partial results are still reported
    except (grader.GradingError, ValueError) as e:
        out["failures"].append(f"{type(e).__name__}: {e}")
    out["sample"] = sample[sample["question_id"].isin(questions)]
    out["rubrics"] = {q: rubrics.load_rubric(pilot._rubric_path(paths, q))
                      for q in questions if pilot._rubric_path(paths, q).exists()}
    out["grades"] = pilot._graded(paths).set_index("item_id")
    check_file = paths.data / "check_grades.csv"
    out["check"] = pd.read_csv(check_file, dtype={"item_id": str}).set_index("item_id") if check_file.exists() else None
    return out


def _mark(found: Optional[bool]) -> str:
    return "—" if found is None else "yes" if found else "**NO**"


def render(out: Dict) -> str:
    sample, grades = out["sample"], out["grades"]
    dev = sample[(sample["split"] == "dev") & sample["item_id"].isin(grades.index)].sort_values("item_id")
    graded_all = sample["item_id"].isin(grades.index).sum()
    kp_rej = sum(len(json.loads(grades.loc[i, "key_points_rejected"])) for i in dev["item_id"])
    m_rej = sum(len(json.loads(grades.loc[i, "misconceptions_rejected"])) for i in dev["item_id"])
    lines = ["# Smoke test: rubric batch 0 (dev answers shown)", ""]
    if out["stopped"]:
        lines += [f"**Partial: stopped before the end ({out['stopped']}).** Re-run to continue from the cache.", ""]
    lines += [
        f"- Questions: {len(out['questions'])}; rubrics made: {len(out['rubrics'])}",
        f"- Answers graded in these questions' calls: {graded_all} of {len(sample)} "
        f"({len(dev)} dev shown; test grades stored, not shown)",
        f"- Replies that failed to parse or validate: {len(out['failures'])}" + "".join(f"\n  - {f}" for f in out["failures"]),
        f"- Dev answers: key-point marks set to 0 for lack of a verbatim quote: {kp_rej}; "
        f"misconception flags not counted for lack of a verbatim quote: {m_rej}",
        "",
    ]
    for qid, r in out["rubrics"].items():
        q = sample[sample["question_id"] == qid].iloc[0]
        lines += [f"## Rubric {qid}", "", f"**Question:** {q['question']}", "", f"**Reference:** {q['reference']}", ""]
        lines += [f"- {k['id']}: {k['text']}" for k in r["key_points"]]
        lines += [f"- {m['id']} (misconception): {m['text']}" for m in r["misconceptions"]]
        lines.append("")
    lines += ["## Dev gradings", "", "| Answer | Marks (K1..) | Misconceptions | Rubric score | score_avg (0-5) |",
              "|---|---|---|---|---|"]
    for row in dev.itertuples():
        g = grades.loc[row.item_id]
        lines.append(f"| {row.item_id} | {json.loads(g['key_point_scores'])} | {json.loads(g['misconceptions_present']) or '—'} "
                     f"| {float(g['rubric_score']):.1f} | {row.score:g} |")
    lines.append("")
    for row in dev.itertuples():
        g = grades.loc[row.item_id]
        rubric = out["rubrics"][row.question_id]
        evidence = json.loads(g["evidence"])
        raw = json.loads(g["key_point_scores_raw"])
        marks = json.loads(g["key_point_scores"])
        rejected_m = json.loads(g["misconceptions_rejected"])
        lines += [f"### {row.item_id} (batch {g['batch_label']} of {int(g['batch_size'])})", "", f"**Answer:** {row.answer}", "",
                  f"**Rubric score:** {float(g['rubric_score']):.1f} / 100 · **human score_avg:** {row.score:g} / 5", "",
                  "| Item | Mark | Evidence | Verbatim in answer |", "|---|---|---|---|"]
        for k, m_raw, m in zip(rubric["key_points"], raw, marks):
            shown = f"{m}" if m == m_raw else f"{m} (model gave {m_raw})"
            lines.append(f"| {k['id']} | {shown} | {evidence.get(k['id']) or '—'} | {_mark(evidence_in_answer(evidence.get(k['id'], ''), row.answer))} |")
        for m in rubric["misconceptions"]:
            state = ("present" if m["id"] in json.loads(g["misconceptions_present"])
                     else "flagged, not counted" if m["id"] in rejected_m else "no")
            lines.append(f"| {m['id']} | {state} | {evidence.get(m['id']) or '—'} | {_mark(evidence_in_answer(evidence.get(m['id'], ''), row.answer))} |")
        lines.append("")
    if out["check"] is not None:
        lines += ["## Batching check (answers in these questions)", "", "| Answer | Batched | Single |", "|---|---|---|"]
        for item_id, c in out["check"].iterrows():
            if item_id in grades.index and item_id in set(sample["item_id"]):
                lines.append(f"| {item_id} | {float(grades.loc[item_id, 'rubric_score']):.1f} | {float(c['rubric_score']):.1f} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test on rubric batch 0.")
    parser.add_argument("--dataset", required=True, choices=sorted(datasets.LOADERS))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--questions", type=int, help="grade only the first N questions of the batch")
    parser.add_argument("--max-calls", type=int, default=0)
    parser.add_argument("--min-interval", type=float, default=4.0)
    args = parser.parse_args(argv)
    df = datasets.load(args.dataset, args.data)
    paths = pilot.Paths(args.dataset)
    llm = CachedLLM(pilot.CACHE_DIR, max_calls=args.max_calls, min_interval_s=args.min_interval)
    out = run(df, paths, llm, args.questions)
    print(f"LLM: {llm.calls} new calls, {llm.hits} from cache, {llm.retries} retries")
    tokens = llm.tokens()
    print("Tokens (new calls): " + ", ".join(f"{k.replace('_token_count', '')} {v}" for k, v in tokens.items()))
    if out["stopped"]:
        print(f"Stopped: {out['stopped']}", file=sys.stderr)
    target = paths.data / "smoke.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(out), encoding="utf-8")
    print(f"Report: {target}")
    return 2 if out["stopped"] else 1 if out["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
