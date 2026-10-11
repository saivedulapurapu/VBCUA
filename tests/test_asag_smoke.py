"""Smoke test helper: dev-only selection, evidence-quote check, report."""
import numpy as np

from research.asag import datasets, llm_cache, pilot, smoke


def test_evidence_quote_check():
    answer = "A stack is  Last-In First-Out. You push and pop at the top."
    assert smoke.evidence_in_answer("last-in first-out", answer) is True
    assert smoke.evidence_in_answer('"push and pop at the top."', answer) is True
    assert smoke.evidence_in_answer("A stack is ... pop at the top", answer) is False  # verbatim only
    assert smoke.evidence_in_answer("first in first out", answer) is False
    assert smoke.evidence_in_answer("", answer) is None


def _scored(n_questions=20):
    df = datasets.make_synthetic(n_questions=n_questions, answers_per_question=40, seed=4)
    rng = np.random.default_rng(0)
    g = rng.uniform(0, 5, len(df))
    return datasets.validate(df.assign(score=g, score_grader_1=g, score_grader_2=g))


def test_smoke_uses_dev_only_and_reuses_pipeline_files(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(pilot, "RESULTS_ROOT", tmp_path / "results")
    df = _scored()
    paths = pilot.Paths("synthetic")
    pilot.stage_split(df, paths)

    def call(prompt, *, schema=None, system=None, temperature=None, model=None):
        if "Student answer" not in prompt:
            return pilot.offline_call(prompt)
        n_kp = prompt.count("\n  K")
        answer = prompt.split("<answer>\n")[1].split("\n</answer>")[0]
        return {"key_points": [{"id": f"K{i}", "evidence": answer.split()[0], "mark": 2} for i in range(1, n_kp + 1)],
                "misconceptions": [{"id": "M1", "evidence": "not in the answer", "present": False}]}

    llm = llm_cache.CachedLLM(tmp_path / "cache", model="m1", max_calls=100, call=call)
    out = smoke.run(df, paths, llm, n_questions=3)
    assert (out["items"]["split"] == "dev").all() and out["items"]["question_id"].nunique() == 3
    assert len(out["rubrics"]) == 3 and len(out["results"]) == len(out["items"])
    assert all((paths.rubrics / f"{q}.json").exists() for q in out["rubrics"])  # reused by the rubrics stage
    report = smoke.render(out, show=2)
    assert "failed to parse or validate: 0" in report
    assert f"not found in the answer: {len(out['results'])}" in report  # the M1 quotes are invented
    assert report.count("## Grading ") == 2 and "## All gradings" in report
    # Same prompts as the pipeline: running the stages afterwards needs no new calls for these
    calls = llm.calls
    assert smoke.run(df, paths, llm, n_questions=3) and llm.calls == calls


def test_smoke_reports_partial_results_when_budget_runs_out(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(pilot, "RESULTS_ROOT", tmp_path / "results")
    df = _scored()
    paths = pilot.Paths("synthetic")
    pilot.stage_split(df, paths)
    llm = llm_cache.CachedLLM(tmp_path / "cache", model="m1", max_calls=2, call=pilot.offline_call)
    out = smoke.run(df, paths, llm, n_questions=3)
    assert "budget" in out["stopped"] and len(out["rubrics"]) == 2 and out["results"] == []
    assert smoke.render(out, show=5).count("## Rubric ") == 2
