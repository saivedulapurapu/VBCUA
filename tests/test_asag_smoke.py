"""Smoke test of the batched design: batch-0 questions only, dev answers shown, partial runs reported."""
import re

import numpy as np

from research.asag import datasets, llm_cache, pilot, smoke


def test_evidence_quote_check():
    answer = "A stack is  Last-In First-Out. You push and pop at the top."
    assert smoke.evidence_in_answer("last-in first-out", answer) is True
    assert smoke.evidence_in_answer('"push and pop at the top."', answer) is True
    assert smoke.evidence_in_answer("A stack is ... pop at the top", answer) is False  # verbatim only
    assert smoke.evidence_in_answer("first in first out", answer) is False
    assert smoke.evidence_in_answer("", answer) is None


def _scored():
    df = datasets.make_synthetic(n_questions=30, answers_per_question=40, seed=4)
    g = np.random.default_rng(0).uniform(0, 5, len(df))
    return datasets.validate(df.assign(score=g, score_grader_1=g, score_grader_2=g))


def fake_call(prompt, *, schema=None, system=None, temperature=None, model=None):
    if "<answer id=" in prompt:
        n_kp = len(re.findall(r"^  K\d+:", prompt, flags=re.M))
        return [{"answer": label,
                 "key_points": [{"id": f"K{i}", "evidence": text.split()[0], "mark": 2} for i in range(1, n_kp + 1)],
                 "misconceptions": [{"id": "M1", "evidence": "not in the answer", "present": True}]}
                for label, text in re.findall(r'<answer id="(A\d+)">\n(.*?)\n</answer>', prompt)]
    if "<answer>" in prompt:
        n_kp = len(re.findall(r"^  K\d+:", prompt, flags=re.M))
        text = prompt.split("<answer>\n")[1].split("\n</answer>")[0]
        return {"key_points": [{"id": f"K{i}", "evidence": text.split()[0], "mark": 2} for i in range(1, n_kp + 1)],
                "misconceptions": [{"id": "M1", "evidence": "", "present": False}]}
    return pilot.offline_call(prompt)


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(pilot, "RESULTS_ROOT", tmp_path / "results")
    df = _scored()
    paths = pilot.Paths("synthetic")
    pilot.stage_split(df, paths)
    return df, paths


def test_smoke_covers_first_rubric_batch_and_shows_dev_only(tmp_path, monkeypatch):
    df, paths = _setup(tmp_path, monkeypatch)
    llm = llm_cache.CachedLLM(tmp_path / "cache", model="m1", max_calls=100, call=fake_call)
    out = smoke.run(df, paths, llm)
    batch0 = smoke.smoke_questions(pilot._sample(df, paths))
    assert out["questions"] == batch0 and len(batch0) == 10 and out["stopped"] is None
    assert set(out["rubrics"]) == set(batch0)
    n_checks = len(set(pilot.check_ids(pilot._sample(df, paths), paths)) & set(out["sample"]["item_id"]))
    assert llm.calls == 1 + len(batch0) + n_checks
    report = smoke.render(out)
    test_ids = out["sample"].loc[out["sample"]["split"] == "test", "item_id"]
    assert not any(f"| {i} |" in report or f"### {i} " in report for i in test_ids)  # no test answers shown
    assert "failed to parse or validate: 0" in report and "misconception flags not counted" in report
    calls = llm.calls
    smoke.run(df, paths, llm)
    assert llm.calls == calls  # everything reused from the cache and the saved files


def test_smoke_reports_partial_results_when_budget_runs_out(tmp_path, monkeypatch):
    df, paths = _setup(tmp_path, monkeypatch)
    llm = llm_cache.CachedLLM(tmp_path / "cache", model="m1", max_calls=3, call=fake_call)
    out = smoke.run(df, paths, llm)
    assert "budget" in out["stopped"] and len(out["rubrics"]) == 10
    graded_questions = out["sample"][out["sample"]["item_id"].isin(out["grades"].index)]["question_id"].nunique()
    assert graded_questions == 2
    assert smoke.render(out).count("## Rubric ") == 10
