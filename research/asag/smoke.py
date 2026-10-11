"""
Smoke test of rubric generation and grading on a few DEV answers (never test answers).

Picks the dev questions with the most dev answers (ties: question id), generates their rubrics
exactly as the `rubrics` stage does (same prompt, same cache, same saved file, so nothing is
paid twice), grades their dev answers exactly as the `grade` stage does, and checks that every
reply parsed and that every evidence quote really occurs in the answer.

    python -m research.asag.smoke --dataset mohler --data research/data/mohler --questions 6 --max-calls 18

The report (with answer text) goes to research/data/<dataset>/smoke.md, which is git-ignored.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from research.asag import datasets, grader, pilot, rubrics
from research.asag.llm_cache import BudgetExceeded, CachedLLM, QuotaStop


def _norm(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def evidence_in_answer(quote: str, answer: str) -> Optional[bool]:
    """
    True if the quote occurs in the answer (case and whitespace ignored; parts joined by "..."
    must appear in order), False if not, None for an empty quote (nothing claimed).
    """
    parts = [p.strip(" \"'.,;:") for p in re.split(r"\.\.\.|…", _norm(quote))]
    parts = [p for p in parts if p]
    if not parts:
        return None
    haystack, start = _norm(answer), 0
    for part in parts:
        found = haystack.find(part, start)
        if found < 0:
            return False
        start = found + len(part)
    return True


def select_dev(sample: pd.DataFrame, n_questions: int) -> pd.DataFrame:
    dev = sample[sample["split"] == "dev"]
    counts = dev.groupby("question_id").size().reset_index(name="n")
    chosen = counts.sort_values(["n", "question_id"], ascending=[False, True]).head(n_questions)["question_id"]
    return dev[dev["question_id"].isin(set(chosen))].sort_values("item_id")


def run(df: pd.DataFrame, paths: "pilot.Paths", llm: CachedLLM, n_questions: int) -> Dict:
    items = select_dev(pilot._sample(df, paths), n_questions)
    assert (items["split"] == "dev").all()
    rubric_by_q, results, failures = {}, [], []
    out = {"items": items, "rubrics": rubric_by_q, "results": results, "failures": failures, "stopped": None}
    try:
        _run(items, paths, llm, out)
    except (BudgetExceeded, QuotaStop) as e:
        out["stopped"] = str(e)  # partial results are still reported
    return out


def _run(items: pd.DataFrame, paths: "pilot.Paths", llm: CachedLLM, out: Dict) -> None:
    rubric_by_q, results, failures = out["rubrics"], out["results"], out["failures"]
    for q in pilot._questions(items).itertuples():
        path = paths.rubrics / f"{pilot._safe(q.question_id)}.json"
        if not path.exists():
            try:
                rubric = rubrics.generate_rubric(q.question, q.reference, llm)
            except (BudgetExceeded, QuotaStop):
                raise
            except Exception as e:
                failures.append(f"rubric {q.question_id}: {type(e).__name__}: {e}")
                continue
            key = llm.key(rubrics.generation_prompt(q.question, q.reference), rubrics.GENERATION_SCHEMA, rubrics.GENERATION_SYSTEM)
            rubrics.save_rubric(path, q.question_id, rubric, llm.model, key)
        rubric_by_q[q.question_id] = rubrics.load_rubric(path)
    for row in items.itertuples():
        if row.question_id not in rubric_by_q:
            continue
        try:
            result = grader.grade_answer(row.question, row.answer, rubric_by_q[row.question_id], llm)
        except (BudgetExceeded, QuotaStop):
            raise
        except Exception as e:
            failures.append(f"grade {row.item_id}: {type(e).__name__}: {e}")
            continue
        checks = {i: evidence_in_answer(quote, row.answer) for i, quote in result["evidence"].items()}
        results.append({"row": row, "result": result, "checks": checks})


def render(out: Dict, show: int) -> str:
    lines = ["# Smoke test (dev answers only)", ""]
    if out.get("stopped"):
        lines += [f"**Partial: stopped before the end ({out['stopped']}).** Re-run to continue from the cache.", ""]
    n_quotes = sum(1 for r in out["results"] for v in r["checks"].values() if v is not None)
    missing = [(r["row"].item_id, i) for r in out["results"] for i, v in r["checks"].items() if v is False]
    lines += [
        f"- Rubrics: {len(out['rubrics'])}; gradings: {len(out['results'])} of {len(out['items'])} dev answers",
        f"- Replies that failed to parse or validate: {len(out['failures'])}" + "".join(f"\n  - {f}" for f in out["failures"]),
        f"- Evidence quotes checked: {n_quotes}; not found in the answer: {len(missing)}"
        + "".join(f"\n  - {item} {i}" for item, i in missing),
        "",
    ]
    for qid, r in out["rubrics"].items():
        q = out["items"][out["items"]["question_id"] == qid].iloc[0]
        lines += [f"## Rubric {qid}", "", f"**Question:** {q['question']}", "", f"**Reference:** {q['reference']}", ""]
        lines += [f"- {k['id']}: {k['text']}" for k in r["key_points"]]
        lines += [f"- {m['id']} (misconception): {m['text']}" for m in r["misconceptions"]]
        lines.append("")
    for g in out["results"][:show]:
        row, res = g["row"], g["result"]
        lines += [f"## Grading {row.item_id}", "", f"**Answer:** {row.answer}", "",
                  f"**Rubric score:** {res['score']:.1f} / 100 · **human score_avg:** {row.score:g} / 5 (label: {row.label})", "",
                  "| Item | Mark | Evidence | Quote found |", "|---|---|---|---|"]
        rubric = out["rubrics"][row.question_id]
        for k, mark in zip(rubric["key_points"], res["key_point_scores"]):
            lines.append(f"| {k['id']} | {mark} | {res['evidence'][k['id']] or '—'} | {_mark(g['checks'][k['id']])} |")
        for m in rubric["misconceptions"]:
            present = "present" if m["id"] in res["misconceptions_present"] else "no"
            lines.append(f"| {m['id']} | {present} | {res['evidence'][m['id']] or '—'} | {_mark(g['checks'][m['id']])} |")
        lines.append("")
    if len(out["results"]) > show:
        lines += ["## All gradings", "", "| Item | Marks | Misconceptions | Rubric score | score_avg |", "|---|---|---|---|---|"]
        for g in out["results"]:
            lines.append(f"| {g['row'].item_id} | {g['result']['key_point_scores']} | {g['result']['misconceptions_present'] or '—'} "
                         f"| {g['result']['score']:.1f} | {g['row'].score:g} |")
    return "\n".join(lines) + "\n"


def _mark(found: Optional[bool]) -> str:
    return "—" if found is None else "yes" if found else "**NO**"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test on a few dev answers.")
    parser.add_argument("--dataset", required=True, choices=sorted(datasets.LOADERS))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument("--show", type=int, default=5, help="gradings shown in full")
    parser.add_argument("--max-calls", type=int, default=0)
    parser.add_argument("--min-interval", type=float, default=4.0)
    args = parser.parse_args(argv)
    df = datasets.load(args.dataset, args.data)
    paths = pilot.Paths(args.dataset)
    llm = CachedLLM(pilot.CACHE_DIR, max_calls=args.max_calls, min_interval_s=args.min_interval)
    out = run(df, paths, llm, args.questions)
    print(f"LLM: {llm.calls} new calls, {llm.hits} from cache, {llm.retries} retries")
    if out["stopped"]:
        print(f"Stopped: {out['stopped']}", file=sys.stderr)
    report = render(out, args.show)
    target = paths.data / "smoke.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    print(f"Report: {target}")
    return 2 if out["stopped"] else 0 if not out["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
