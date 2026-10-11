"""Rubric grader: prompts, verbatim-evidence rule, score rule, batching, caching, pilot wiring."""
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
ANSWER = "A stack is last in first out. You push and pop at the top."


def reply(marks=(2, 1), present=(False,), kp_evidence=("last in first out", "push and pop"), m_evidence=("",)):
    return {
        "key_points": [{"id": f"K{i}", "evidence": e, "mark": m} for i, (m, e) in enumerate(zip(marks, kp_evidence), 1)],
        "misconceptions": [{"id": f"M{i}", "evidence": e, "present": p} for i, (p, e) in enumerate(zip(present, m_evidence), 1)],
    }


class Fake:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def __call__(self, prompt, *, schema=None, system=None, temperature=None, model=None):
        self.prompts.append({"prompt": prompt, "schema": schema, "system": system, "temperature": temperature})
        return self.answer(prompt) if callable(self.answer) else self.answer


def llm_with(tmp_path, answer, max_calls=5):
    fake = Fake(answer)
    return llm_cache.CachedLLM(tmp_path, model="m1", call=fake, max_calls=max_calls), fake


# ---------------------------------------------------------------- single answer
def test_prompt_has_rubric_and_answer_but_no_reference():
    prompt = grader.build_prompt("What is a stack?", "it is last in first out", RUBRIC)
    for text in ("What is a stack?", "K1: LIFO order", "K2: push and pop at the top", "M1: A stack is FIFO",
                 "<answer>\nit is last in first out\n</answer>", "(1 if the student says items are 'stacked'",
                 "copied word for word"):
        assert text in prompt
    assert "Reference" not in prompt and "label" not in prompt.lower()


def test_grade_answer_parses_marks_and_scores(tmp_path):
    llm, fake = llm_with(tmp_path, reply((2, 1)))
    result = grader.grade_answer("What is a stack?", ANSWER, RUBRIC, llm)
    assert result["key_point_scores"] == [2, 1] and result["key_points_rejected"] == []
    assert result["misconceptions_present"] == [] and result["score"] == 75.0
    assert result["evidence"] == {"K1": "last in first out", "K2": "push and pop", "M1": ""}
    assert fake.prompts[0]["temperature"] == 0.0 and fake.prompts[0]["schema"] == grader.SCHEMA
    assert fake.prompts[0]["system"] == grader.SYSTEM


def test_key_point_without_verbatim_quote_scores_zero(tmp_path):
    llm, _ = llm_with(tmp_path, reply((2, 2), kp_evidence=("last in first out", "items go on top")))
    result = grader.grade_answer("Q", ANSWER, RUBRIC, llm)
    assert result["key_point_scores_raw"] == [2, 2] and result["key_point_scores"] == [2, 0]
    assert result["key_points_rejected"] == ["K2"] and result["score"] == 50.0


def test_zero_mark_needs_no_quote(tmp_path):
    llm, _ = llm_with(tmp_path, reply((0, 2), kp_evidence=("", "push and pop")))
    assert grader.grade_answer("Q", ANSWER, RUBRIC, llm)["key_points_rejected"] == []


def test_misconception_deducts_one_mark(tmp_path):
    answer = "A stack is first in, first out. You push and pop at the top."
    llm, _ = llm_with(tmp_path, reply((0, 2), (True,), kp_evidence=("", "push and pop"),
                                      m_evidence=("stack is first in, first out",)))
    result = grader.grade_answer("Q", answer, RUBRIC, llm)
    assert result["misconceptions_present"] == ["M1"] and result["misconceptions_rejected"] == []
    assert result["score"] == 25.0


@pytest.mark.parametrize("evidence", ["", "a stack is FIFO", "stack is ... first out"])
def test_misconception_without_verbatim_quote_is_absent(tmp_path, evidence):
    llm, _ = llm_with(tmp_path, reply((2, 2), (True,), m_evidence=(evidence,)))
    result = grader.grade_answer("Q", ANSWER, RUBRIC, llm)
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
        grader.grade_answer("Q", ANSWER, RUBRIC, llm)


def test_grading_is_cached(tmp_path):
    llm, fake = llm_with(tmp_path, reply())
    assert not grader.is_cached("Q", ANSWER, RUBRIC, llm)
    first = grader.grade_answer("Q", ANSWER, RUBRIC, llm)
    assert grader.is_cached("Q", ANSWER, RUBRIC, llm)
    assert grader.grade_answer("Q", ANSWER, RUBRIC, llm) == first and len(fake.prompts) == 1


def test_grade_explanation_with_concept_rubric(tmp_path):
    rubric = next(r for r in concept_rubrics.load_all() if r["id"] == "deadlock")
    n_kp, n_m = len(rubric["key_points"]), len(rubric["misconceptions"])
    transcript = "um so a deadlock is when two threads wait on each other"
    llm, fake = llm_with(tmp_path, reply([2] * n_kp, [False] * n_m, kp_evidence=["two threads wait"] * n_kp,
                                         m_evidence=[""] * n_m))
    result = grader.grade_explanation(transcript, rubric, llm)
    assert result["score"] == 100.0
    prompt = fake.prompts[0]["prompt"]
    assert 'Explain the concept "Deadlock"' in prompt and "Coffman" in prompt and "transcript" in prompt


# ---------------------------------------------------------------- batches
ANSWERS = {"q.0": "last in first out", "q.1": "push and pop at the top", "q.2": "no idea", "q.3": "a queue"}


def test_batch_order_is_fixed_shuffle():
    order = grader.batch_order(list(ANSWERS), "q")
    assert sorted(order) == sorted(ANSWERS) and order == grader.batch_order(list(reversed(ANSWERS)), "q")
    many = [f"x.{i}" for i in range(12)]
    assert grader.batch_order(many, "x") != sorted(many)  # actually shuffled
    assert grader.batch_order(many, "x") != grader.batch_order(many, "y")  # per question


def test_batch_prompt_labels_answers_and_hides_ids():
    order = grader.batch_order(list(ANSWERS), "q")
    prompt = grader.build_batch_prompt("What is a stack?", [ANSWERS[i] for i in order], RUBRIC)
    assert "Grade each answer independently against the rubric; do not compare answers." in prompt
    for n, item_id in enumerate(order, 1):
        assert f'<answer id="A{n}">\n{ANSWERS[item_id]}\n</answer>' in prompt
    assert "q.0" not in prompt and "Reference" not in prompt
    assert "K1: LIFO order" in prompt and "M1: A stack is FIFO" in prompt and "copied word for word" in prompt


def _batch_reply(prompt):
    """Gives each answer K1=2 quoting its first two words, K2=0, and flags M1 with a made-up quote."""
    out = []
    for label, text in __import__("re").findall(r'<answer id="(A\d+)">\n(.*?)\n</answer>', prompt):
        quote = " ".join(text.split()[:2])
        out.append({"answer": label,
                    "key_points": [{"id": "K1", "evidence": quote, "mark": 2}, {"id": "K2", "evidence": "", "mark": 0}],
                    "misconceptions": [{"id": "M1", "evidence": "invented words", "present": True}]})
    return list(reversed(out))  # order of the reply must not matter


def test_grade_batch_maps_labels_back_to_items(tmp_path):
    llm, fake = llm_with(tmp_path, _batch_reply)
    results = grader.grade_batch("What is a stack?", "q", ANSWERS, RUBRIC, llm)
    assert set(results) == set(ANSWERS) and len(fake.prompts) == 1
    order = grader.batch_order(list(ANSWERS), "q")
    for n, item_id in enumerate(order, 1):
        r = results[item_id]
        assert r["batch_label"] == f"A{n}" and r["batch_size"] == 4
        assert r["evidence"]["K1"] == " ".join(ANSWERS[item_id].split()[:2])  # its own answer's quote
        assert r["key_point_scores"] == [2, 0] and r["misconceptions_rejected"] == ["M1"] and r["score"] == 50.0
    assert grader.is_batch_cached("What is a stack?", "q", ANSWERS, RUBRIC, llm)


def test_grade_batch_quote_from_another_answer_is_rejected(tmp_path):
    def swapped(prompt):
        data = _batch_reply(prompt)
        quotes = [d["key_points"][0]["evidence"] for d in data]
        for d, q in zip(data, quotes[1:] + quotes[:1]):
            d["key_points"][0]["evidence"] = q
        return data

    llm, _ = llm_with(tmp_path, swapped)
    results = grader.grade_batch("Q", "q", ANSWERS, RUBRIC, llm)
    assert all(r["key_points_rejected"] == ["K1"] and r["score"] == 0.0 for r in results.values())


@pytest.mark.parametrize("change", [lambda d: d[:-1], lambda d: d + d[:1], lambda d: {"not": "a list"}])
def test_grade_batch_rejects_missing_or_extra_labels(tmp_path, change):
    llm, _ = llm_with(tmp_path, lambda p: change(_batch_reply(p)))
    with pytest.raises(grader.GradingError):
        grader.grade_batch("Q", "q", ANSWERS, RUBRIC, llm)


# ---------------------------------------------------------------- pilot wiring
def test_pilot_stages_use_batches_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(pilot, "RESULTS_ROOT", tmp_path / "results")
    df = datasets.make_synthetic(n_questions=30, answers_per_question=12, seed=1)
    paths = pilot.Paths("synthetic")
    pilot.stage_split(df, paths)
    n_questions = pilot._sample(df, paths)["question_id"].nunique()

    def fake_call(prompt, *, schema=None, system=None, temperature=None, model=None):
        return _batch_reply(prompt) if "<answer id=" in prompt else pilot.offline_call(prompt)

    def rubric_two_points(prompt, **kw):
        data = pilot.offline_call(prompt)
        for r in data:
            r["key_points"] = [{"id": "K1", "text": "a"}, {"id": "K2", "text": "b"}]
        return data

    rubric_llm = llm_cache.CachedLLM(tmp_path / "cache", model="m1", max_calls=10, call=rubric_two_points)
    pilot.stage_rubrics(df, paths, rubric_llm)
    assert rubric_llm.calls == -(-n_questions // 10)  # one call per batch of up to 10

    llm = llm_cache.CachedLLM(tmp_path / "cache", model="m1", max_calls=5, call=fake_call)
    with pytest.raises(llm_cache.BudgetExceeded):
        pilot.stage_grade(df, paths, llm, offline=False)
    graded = pd.read_csv(paths.grades)
    assert graded["item_id"].map(lambda i: i.split(".")[0]).nunique() == 5  # 5 questions, all their answers
    assert pilot.stage_plan(df, paths, llm, offline=False)["grading_calls"] == n_questions - 5

    llm = llm_cache.CachedLLM(tmp_path / "cache", model="m1", max_calls=n_questions, call=fake_call)
    grades = pilot.stage_grade(df, paths, llm, offline=False)
    assert len(grades) == 300 and llm.calls == n_questions - 5
    assert set(grades["rubric_score"]) == {50.0} and json.loads(grades["misconceptions_rejected"][0]) == ["M1"]

    # Batching check: 10 fixed dev answers, one call each
    def single(prompt, **kw):
        text = prompt.split("<answer>\n")[1].split("\n</answer>")[0]
        return {"key_points": [{"id": "K1", "evidence": text.split()[0], "mark": 1}, {"id": "K2", "evidence": "", "mark": 0}],
                "misconceptions": [{"id": "M1", "evidence": "", "present": False}]}

    llm = llm_cache.CachedLLM(tmp_path / "cache", model="m1", max_calls=10, call=single)
    table = pilot.stage_check(df, paths, llm)
    assert len(table) == 10 and llm.calls == 10 and set(table["difference"]) == {25.0}  # batched 50, single 25
    ids = pd.read_csv(paths.results / "batch_check_ids.csv")["item_id"]
    assert set(pilot._sample(df, paths).set_index("item_id").loc[ids, "split"]) == {"dev"}
    md = (paths.results / f"batch_check_{pilot.GRADER_VERSION}.md").read_text(encoding="utf-8")
    assert "Mean absolute score difference: 25.0" in md
    assert pilot.stage_plan(df, paths, llm, offline=False)["total_calls"] == 0


def test_prompt_v2_counts_paraphrases_and_versions_the_cache(tmp_path):
    assert grader.PROMPT_VERSION == "v2"
    prompt = grader.build_prompt("Q", "A", RUBRIC)
    assert "in any wording (a paraphrase counts)" in prompt and "absent or wrong" in prompt
    assert "quote the student's own paraphrasing words" in prompt
    llm, _ = llm_with(tmp_path, reply())
    grader.grade_answer("Q", ANSWER, RUBRIC, llm)
    key_v2 = llm.key(grader.build_prompt("Q", ANSWER, RUBRIC), grader.SCHEMA, grader.SYSTEM, "v2")
    assert (tmp_path / "m1" / f"{key_v2}.json").exists()
