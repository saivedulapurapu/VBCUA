"""Rubric grader: prompt contents, parsing and validation of marks, score rule, caching, pilot wiring."""
import json

import pandas as pd
import pytest

from research.asag import concept_rubrics, datasets, grader, llm_cache, pilot

RUBRIC = {
    "key_points": [
        {"id": "K1", "text": "LIFO order", "partial_if": "says items are 'stacked' without the order"},
        {"id": "K2", "text": "push and pop at the top"},
    ],
    "misconceptions": [{"id": "M1", "text": "A stack is FIFO"}],
}


def reply(marks=(2, 1), present=(False,)):
    return {
        "key_points": [{"id": f"K{i}", "evidence": "q", "mark": m} for i, m in enumerate(marks, 1)],
        "misconceptions": [{"id": f"M{i}", "evidence": "", "present": p} for i, p in enumerate(present, 1)],
    }


class Fake:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def __call__(self, prompt, *, schema=None, system=None, temperature=None, model=None):
        self.prompts.append({"prompt": prompt, "schema": schema, "system": system, "temperature": temperature})
        return self.answer


def llm_with(tmp_path, answer, max_calls=5):
    fake = Fake(answer)
    return llm_cache.CachedLLM(tmp_path, model="m1", call=fake, max_calls=max_calls), fake


def test_prompt_has_rubric_and_answer_but_no_reference():
    prompt = grader.build_prompt("What is a stack?", "it is last in first out", RUBRIC)
    for text in ("What is a stack?", "K1: LIFO order", "K2: push and pop at the top", "M1: A stack is FIFO",
                 "<answer>\nit is last in first out\n</answer>", "(1 if the student says items are 'stacked'"):
        assert text in prompt
    assert "Reference" not in prompt and "label" not in prompt.lower()


def test_grade_answer_parses_marks_and_scores(tmp_path):
    llm, fake = llm_with(tmp_path, reply((2, 1), (False,)))
    result = grader.grade_answer("What is a stack?", "last in first out", RUBRIC, llm)
    assert result["key_point_scores"] == [2, 1] and result["misconceptions_present"] == []
    assert result["score"] == 75.0
    assert result["evidence"] == {"K1": "q", "K2": "q", "M1": ""}
    assert fake.prompts[0]["temperature"] == 0.0 and fake.prompts[0]["schema"] == grader.SCHEMA
    assert fake.prompts[0]["system"] == grader.SYSTEM


def test_misconception_deducts_one_mark(tmp_path):
    answer = "A stack is first in, first out."
    data = reply((2, 2), (True,))
    data["misconceptions"][0]["evidence"] = "stack is first in, first out"
    llm, _ = llm_with(tmp_path, data)
    result = grader.grade_answer("Q", answer, RUBRIC, llm)
    assert result["misconceptions_present"] == ["M1"] and result["misconceptions_rejected"] == []
    assert result["score"] == 75.0


@pytest.mark.parametrize("evidence", ["", "a stack is FIFO", "stack is first in ... first out"])
def test_misconception_without_verbatim_quote_is_absent(tmp_path, evidence):
    data = reply((2, 2), (True,))
    data["misconceptions"][0]["evidence"] = evidence
    llm, _ = llm_with(tmp_path, data)
    result = grader.grade_answer("Q", "A stack is first in, first out.", RUBRIC, llm)
    assert result["misconceptions_present"] == [] and result["misconceptions_rejected"] == ["M1"]
    assert result["score"] == 100.0  # not deducted


def test_quote_in_answer_is_verbatim():
    answer = "It is  Last-In First-Out, so you\npush ‘on top’."
    assert grader.quote_in_answer("last-in first-out", answer)
    assert grader.quote_in_answer("\"you push 'on top'.\"", answer)  # case, spaces, quote styles, framing
    assert not grader.quote_in_answer("last in first out", answer)  # hyphens matter
    assert not grader.quote_in_answer("It is ... on top", answer)  # no ellipsis joins
    assert not grader.quote_in_answer("", answer) and not grader.quote_in_answer("  .", answer)


def test_score_rule():
    assert grader.rubric_score([2, 2, 2], 0) == 100.0
    assert grader.rubric_score([0, 0], 0) == 0.0
    assert grader.rubric_score([1, 0], 3) == 0.0  # never negative
    assert grader.rubric_score([2, 1, 0, 0], 1) == 25.0
    with pytest.raises(ValueError):
        grader.rubric_score([], 0)


def test_rule_shown_to_reviewer_matches_code():
    assert "max(0, sum of key-point marks − number of misconceptions stated)" in concept_rubrics.SCORING_RULE
    assert "(2 × number of key points) × 100" in concept_rubrics.SCORING_RULE


@pytest.mark.parametrize("bad", [
    {**reply(), "key_points": reply()["key_points"][:1]},                                  # missing K2
    {**reply(), "key_points": reply()["key_points"] + reply()["key_points"][:1]},          # duplicate K1
    {**reply(), "misconceptions": []},                                                     # missing M1
    reply((2, 3)),                                                                         # mark out of range
    {**reply(), "key_points": [{"id": "K1", "evidence": "", "mark": True}, reply()["key_points"][1]]},
])
def test_invalid_marks_rejected(tmp_path, bad):
    llm, _ = llm_with(tmp_path, bad)
    with pytest.raises(grader.GradingError):
        grader.grade_answer("Q", "A", RUBRIC, llm)


def test_grading_is_cached(tmp_path):
    llm, fake = llm_with(tmp_path, reply())
    assert not grader.is_cached("Q", "A", RUBRIC, llm)
    first = grader.grade_answer("Q", "A", RUBRIC, llm)
    assert grader.is_cached("Q", "A", RUBRIC, llm)
    assert grader.grade_answer("Q", "A", RUBRIC, llm) == first and len(fake.prompts) == 1


def test_grade_explanation_with_concept_rubric(tmp_path):
    rubric = next(r for r in concept_rubrics.load_all() if r["id"] == "deadlock")
    n_kp, n_m = len(rubric["key_points"]), len(rubric["misconceptions"])
    llm, fake = llm_with(tmp_path, reply([2] * n_kp, [False] * n_m))
    result = grader.grade_explanation("um so a deadlock is when two threads wait on each other", rubric, llm)
    assert result["score"] == 100.0
    prompt = fake.prompts[0]["prompt"]
    assert 'Explain the concept "Deadlock"' in prompt and "Coffman" in prompt and "transcript" in prompt


def test_pilot_grade_stage_uses_the_grader(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(pilot, "RESULTS_ROOT", tmp_path / "results")
    df = datasets.make_synthetic(n_questions=30, answers_per_question=12, seed=1)
    paths = pilot.Paths("synthetic")
    pilot.stage_split(df, paths)
    rubric_llm = llm_cache.CachedLLM(tmp_path / "cache", model="m1", max_calls=10**4, call=pilot.offline_call)
    pilot.stage_rubrics(df, paths, rubric_llm)

    def fake_call(prompt, *, schema=None, system=None, temperature=None, model=None):
        n_kp = prompt.count("\n  K")
        return reply([1] * n_kp, [False])

    llm = llm_cache.CachedLLM(tmp_path / "cache", model="m1", max_calls=10, call=fake_call)
    with pytest.raises(llm_cache.BudgetExceeded):
        pilot.stage_grade(df, paths, llm, offline=False)
    assert len(pd.read_csv(paths.grades)) == 10  # progress kept when the budget runs out
    assert pilot.stage_plan(df, paths, llm, offline=False)["grading_calls"] == 290

    llm = llm_cache.CachedLLM(tmp_path / "cache", model="m1", max_calls=290, call=fake_call)
    grades = pilot.stage_grade(df, paths, llm, offline=False)
    assert len(grades) == 300 and set(grades["rubric_score"]) == {50.0}
    assert json.loads(grades["evidence"][0])["K1"] == "q"
