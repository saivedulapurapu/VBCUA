"""Short-answer grading pilot: loaders, sampling, LLM cache, rubric generation, cut-offs, offline run."""
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from research.asag import datasets, llm_cache, pilot, rubrics, sampling, thresholds


# ---------------------------------------------------------------- datasets
def test_synthetic_has_common_shape():
    df = datasets.make_synthetic()
    assert list(df.columns) == datasets.COLUMNS
    assert set(df["label"]) == set(datasets.LABELS)
    assert df.equals(datasets.make_synthetic())  # deterministic


def test_engsaf_loader_reads_official_columns(tmp_path):
    folder = tmp_path / "engsaf"
    folder.mkdir()
    pd.DataFrame({
        "Question_id": ["7", "7", "8"],
        "Question": ["What is a stack?", "What is a stack?", "What is TCP?"],
        "Student Answer": ["LIFO list", "a queue", "reliable transport"],
        "Correct Answer": ["A LIFO structure", "A LIFO structure", "Reliable, ordered transport"],
        "output_label": ["2", "0", "1"],
        "feedback": ["", "", ""],
    }).to_csv(folder / "train.csv", index=False)
    df = datasets.load("engsaf", folder)
    assert list(df["item_id"]) == ["train:0", "train:1", "train:2"]
    assert list(df["label"]) == ["correct", "incorrect", "partial"]
    assert df.loc[0, "reference"] == "A LIFO structure"


def test_engsaf_loader_explains_unknown_columns(tmp_path):
    csv = tmp_path / "x.csv"
    pd.DataFrame({"q": [1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Pass `columns`"):
        datasets.load_engsaf(csv)


def _write_mohler(root):
    data = root / "data"
    for sub in ("docs", "raw", "scores/1.1", "scores/1.2", "scores/2.1"):
        (data / sub).mkdir(parents=True, exist_ok=True)
    (data / "docs" / "files").write_text("1.1\n#1.2\n2.1\n")
    (data / "raw" / "questions").write_text("1.1 What is a stack?\n1.2 Order these.\n2.1 What is a queue?\n")
    (data / "raw" / "answers").write_text("1.1 A LIFO list.\n1.2 a b c\n2.1 A FIFO list.\n")
    (data / "raw" / "1.1").write_bytes("1.1 last in first out \n1.1 caf\xe9 list\n1.1 no idea\n".encode("latin-1"))
    (data / "raw" / "2.1").write_text("2.1 first in first out\n")
    (data / "scores" / "1.1" / "ave").write_text("5\n3.5\n1\n")
    (data / "scores" / "2.1" / "ave").write_text("4.5\n")


def test_mohler_loader_skips_commented_questions_and_bins_grades(tmp_path):
    _write_mohler(tmp_path)
    df = datasets.load("mohler_raw", tmp_path)
    assert list(df["item_id"]) == ["1.1.0", "1.1.1", "1.1.2", "2.1.0"]
    assert "1.2" not in set(df["question_id"])
    assert list(df["label"]) == ["correct", "partial", "incorrect", "correct"]
    assert df.loc[1, "answer"] == "café list"  # latin-1
    assert df.loc[0, "question"] == "What is a stack?" and df.loc[0, "reference"] == "A LIFO list."


def test_mohler_hf_loader(tmp_path):
    pd.DataFrame({
        "id": ["E01.Q01.A00", "E01.Q01.A01", "E11.Q02.A00"],
        "question": ["What is a stack? <STOP>", "What is a stack? <STOP>", "Define a heap. <STOP>"],
        "instructor_answer": ["A LIFO list. <STOP>", "A LIFO list. <STOP>", "A complete tree. <STOP> Ordered."],
        "student_answer": ["Last in <STOP> first out.", "no idea", "a tree"],
        "score_grader_1": [5.0, 1.0, 8.0],
        "score_grader_2": [4.0, 2.0, 6.0],
        "score_avg": [4.5, 1.5, 3.5],
    }).to_parquet(tmp_path / "cleaned-oe-00001.parquet")
    df = datasets.load("mohler", tmp_path)
    assert list(df["question_id"]) == ["E01.Q01", "E01.Q01", "E11.Q02"]
    assert df.loc[0, "answer"] == "Last in first out." and df.loc[0, "question"] == "What is a stack?"
    assert df.loc[2, "reference"] == "A complete tree. Ordered."
    assert list(df["label"]) == ["correct", "incorrect", "partial"]
    assert list(df["score"]) == [4.5, 1.5, 3.5]
    # Exams E11/E12 were graded 0-10 per grader: halved to 0-5
    assert list(df["score_grader_1"]) == [5.0, 1.0, 4.0] and list(df["score_grader_2"]) == [4.0, 2.0, 3.0]


def test_mohler_parser_tokens_become_brackets():
    clean = datasets._clean_mohler_text
    assert clean("inside a function -LRB- local variables -RRB- or outside <STOP>") == \
        "inside a function (local variables) or outside"
    assert clean("a -LSB- i -RSB- and -LCB- x -RCB-") == "a [i] and {x}"
    assert clean("f -LRB- g -LRB- x -RRB- -RRB-") == "f (g (x))"
    assert not any(t in clean(" ".join(datasets.PTB_BRACKETS)) for t in datasets.PTB_BRACKETS)


def test_mohler_cutoffs():
    assert [datasets.mohler_label(s) for s in (0, 2.49, 2.5, 4.49, 4.5, 5)] == [
        "incorrect", "incorrect", "partial", "partial", "correct", "correct"]


def test_validate_rejects_bad_data():
    df = datasets.make_synthetic(2, 2)
    with pytest.raises(ValueError, match="Unknown labels"):
        datasets.validate(df.assign(label="wrong"))
    with pytest.raises(ValueError, match="unique"):
        datasets.validate(pd.concat([df, df.iloc[:1]]))


# ---------------------------------------------------------------- sampling
@pytest.fixture(scope="module")
def big():
    return datasets.make_synthetic(n_questions=60, answers_per_question=20, seed=3)


def test_split_sizes_and_reproducibility(big):
    split = sampling.make_split(big)
    assert len(split) == 300 and split["item_id"].is_unique
    assert split["split"].value_counts().to_dict() == {"test": 200, "dev": 100}
    assert split.equals(sampling.make_split(big))
    assert not split.equals(sampling.make_split(big, seed=7))
    assert list(split.columns) == ["item_id", "question_id", "split"]  # no text, no labels


def test_split_is_stratified_by_label(big):
    sample = sampling.apply_split(big, sampling.make_split(big))
    overall = big["label"].value_counts(normalize=True)
    for part in (sample, sample[sample["split"] == "dev"]):
        share = part["label"].value_counts(normalize=True)
        for label in datasets.LABELS:
            assert abs(share[label] - overall[label]) < 0.03


def test_stratified_quota_is_proportional():
    df = pd.DataFrame({"item_id": [str(i) for i in range(100)], "question_id": "q",
                       "label": ["correct"] * 70 + ["incorrect"] * 30})
    chosen = set(sampling.stratified_ids(df, 10))
    assert len(chosen) == 10
    assert sum(df.set_index("item_id").loc[sorted(chosen), "label"] == "correct") == 7


def test_apply_split_detects_another_release(big):
    split = sampling.make_split(big)
    with pytest.raises(ValueError, match="not in the dataset"):
        sampling.apply_split(big[big["item_id"] != split["item_id"][0]], split)


# ---------------------------------------------------------------- LLM cache
class FakeCall:
    def __init__(self, errors=()):
        self.errors = list(errors)
        self.calls = []

    def __call__(self, prompt, *, schema=None, system=None, temperature=None, model=None):
        self.calls.append({"prompt": prompt, "temperature": temperature, "model": model})
        if self.errors:
            raise self.errors.pop(0)
        return {"echo": prompt}


class ApiError(Exception):
    def __init__(self, code):
        super().__init__(f"{code} RESOURCE_EXHAUSTED quota")
        self.code = code


def _llm(tmp_path, call, max_calls=10):
    return llm_cache.CachedLLM(tmp_path, model="m1", call=call, max_calls=max_calls, sleep=lambda s: None)


def test_cache_answers_repeat_prompts_from_disk(tmp_path):
    call = FakeCall()
    llm = _llm(tmp_path, call)
    assert llm.generate_json("p") == {"echo": "p"}
    assert llm.generate_json("p") == {"echo": "p"}
    assert len(call.calls) == 1 and call.calls[0]["temperature"] == 0.0 and call.calls[0]["model"] == "m1"
    # A new object reading the same folder needs no call either
    again = llm_cache.CachedLLM(tmp_path, model="m1", call=FakeCall(), max_calls=0)
    assert again.generate_json("p") == {"echo": "p"} and again.hits == 1
    record = json.loads(next((tmp_path / "m1").glob("*.json")).read_text(encoding="utf-8"))
    assert record["prompt"] == "p" and record["model"] == "m1"


def test_cache_key_depends_on_model_prompt_schema_and_system(tmp_path):
    llm = _llm(tmp_path, FakeCall())
    other = llm_cache.CachedLLM(tmp_path, model="m2", call=FakeCall())
    keys = {llm.key("p"), llm.key("q"), llm.key("p", {"type": "object"}), llm.key("p", system="s"), other.key("p")}
    assert len(keys) == 5 and llm.key("p") == llm.key("p")


def test_budget_stops_uncached_calls(tmp_path):
    llm = _llm(tmp_path, FakeCall(), max_calls=1)
    llm.generate_json("a")
    with pytest.raises(llm_cache.BudgetExceeded):
        llm.generate_json("b")
    assert llm.generate_json("a")  # cached answers are always allowed


def test_backoff_on_quota_then_success(tmp_path):
    slept = []
    call = FakeCall([ApiError(429), ApiError(503)])
    llm = llm_cache.CachedLLM(tmp_path, model="m1", call=call, max_calls=1, sleep=slept.append, backoff_s=(1, 2, 3))
    assert llm.generate_json("p") == {"echo": "p"}
    assert slept == [1, 2] and llm.calls == 1 and llm.retries == 2


def test_backoff_gives_up_with_quota_stop(tmp_path):
    error = ApiError(429)
    wrapped = RuntimeError("LLMError")  # as raised by llm_client, with the API error as the cause
    wrapped.__cause__ = error
    call = FakeCall([wrapped] * 4)
    llm = llm_cache.CachedLLM(tmp_path, model="m1", call=call, max_calls=5, sleep=lambda s: None, backoff_s=(1, 2, 3))
    with pytest.raises(llm_cache.QuotaStop):
        llm.generate_json("p")
    assert len(call.calls) == 4 and not list(tmp_path.rglob("*.json"))


def test_other_errors_are_not_retried(tmp_path):
    call = FakeCall([ValueError("bad schema")])
    llm = llm_cache.CachedLLM(tmp_path, model="m1", call=call, max_calls=5, sleep=lambda s: None)
    with pytest.raises(ValueError):
        llm.generate_json("p")
    assert len(call.calls) == 1


# ---------------------------------------------------------------- rubrics
def test_rubric_generation_sees_only_question_and_reference():
    params = list(inspect.signature(rubrics.generate_rubric).parameters)
    assert params == ["question", "reference", "llm"]
    prompt = rubrics.generation_prompt("What is a stack?", "A LIFO structure.")
    assert "What is a stack?" in prompt and "A LIFO structure." in prompt
    assert "Student" not in prompt and "label" not in prompt.lower()


def test_generate_and_save_rubric(tmp_path):
    def call(prompt, **kw):
        return {"key_points": [{"id": "x", "text": " LIFO order "}, {"id": "y", "text": "push and pop"}],
                "misconceptions": [{"id": "z", "text": "A stack is FIFO"}]}
    llm = llm_cache.CachedLLM(tmp_path / "cache", model="m1", call=call, max_calls=1)
    rubric = rubrics.generate_rubric("What is a stack?", "A LIFO structure.", llm)
    assert [k["id"] for k in rubric["key_points"]] == ["K1", "K2"] and rubric["key_points"][0]["text"] == "LIFO order"
    assert rubric["misconceptions"] == [{"id": "M1", "text": "A stack is FIFO"}]
    path = tmp_path / "r" / "q1.json"
    rubrics.save_rubric(path, "q1", rubric, "m1", "abc")
    saved = path.read_text(encoding="utf-8")
    assert "What is a stack?" not in saved and "A LIFO structure." not in saved
    assert rubrics.load_rubric(path)["question_id"] == "q1"


# ---------------------------------------------------------------- cut-offs
def test_apply_cutoffs_boundaries():
    assert list(thresholds.apply([0, 49.9, 50, 79.9, 80, 100], thresholds.APP_CUTOFFS)) == [0, 0, 1, 1, 2, 2]


def test_tune_finds_separating_cutoffs():
    scores = [10, 20, 30, 55, 60, 65, 90, 95]
    labels = ["incorrect"] * 3 + ["partial"] * 3 + ["correct"] * 2
    best = thresholds.tune(scores, labels)
    assert 30 < best["low"] < 55 and 65 < best["high"] < 90
    assert best["dev_qwk"] == pytest.approx(1.0) and best["dev_accuracy"] == 1.0


def test_tune_is_deterministic_with_ties():
    scores, labels = [50, 50, 50], ["partial", "correct", "incorrect"]
    assert thresholds.tune(scores, labels) == thresholds.tune(scores, labels)


# ---------------------------------------------------------------- whole pipeline, offline
def _fake_encode(texts):
    """Bag-of-words vectors, so the run needs no model download."""
    vocab = sorted({w for t in texts for w in pilot._words(t)} | set(datasets._TOPIC_WORDS))
    index = {w: i for i, w in enumerate(vocab)}
    out = np.zeros((len(texts), len(vocab)))
    for row, text in enumerate(texts):
        for w in pilot._words(text):
            out[row, index[w]] += 1
    return out


@pytest.fixture
def offline_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(pilot, "RESULTS_ROOT", tmp_path / "results")
    return pilot.Paths("synthetic")


def test_offline_pipeline_end_to_end(offline_paths, big):
    paths = offline_paths
    llm = llm_cache.CachedLLM(paths.data / "cache", model=pilot.OFFLINE_MODEL, max_calls=10**6, call=pilot.offline_call)
    split = pilot.stage_split(big, paths)
    assert pilot.stage_split(big.sample(frac=1, random_state=0), paths).equals(split)  # never re-drawn

    plan = pilot.stage_plan(big, paths, llm, offline=True)
    assert plan["answers"] == 300 and plan["rubric_calls"] == -(-plan["questions"] // 10)  # batches of up to 10

    assert pilot.stage_rubrics(big, paths, llm) == plan["questions"]
    assert pilot.stage_rubrics(big, paths, llm) == 0
    assert pilot.stage_plan(big, paths, llm, offline=True)["rubric_calls"] == 0

    pilot.stage_grade(big, paths, llm, offline=True)
    table = pilot.stage_report(big, paths, "synthetic", encode=_fake_encode)

    assert list(table["method"]) == pilot.METHODS
    assert (table["n"] == 200).all()  # reported on test only
    cut = json.loads(paths.thresholds.read_text())
    assert cut["fitted_on"] == "dev" and set(cut["label_cutoffs"]) >= {"v1", "b0", "rubric"}
    assert cut["linear_maps_to_gold_scale"] is None  # synthetic data has labels only
    preds = pd.read_csv(paths.predictions)
    assert "answer" not in preds.columns and len(preds) == 300
    md = (paths.test / "metrics.md").read_text(encoding="utf-8")
    assert "n=200" in md and "rubric_tuned" in md


def test_cutoffs_use_dev_only(offline_paths, big, monkeypatch):
    """Changing test labels must not change the tuned cut-offs."""
    paths = offline_paths
    llm = llm_cache.CachedLLM(paths.data / "cache", model=pilot.OFFLINE_MODEL, max_calls=10**6, call=pilot.offline_call)
    pilot.stage_split(big, paths)
    pilot.stage_rubrics(big, paths, llm)
    pilot.stage_grade(big, paths, llm, offline=True)
    pilot.stage_report(big, paths, "synthetic", encode=_fake_encode)
    first = json.loads(paths.thresholds.read_text())

    test_ids = set(sampling.load_split(paths.split).query("split == 'test'")["item_id"])
    flipped = big.copy()
    flipped.loc[flipped["item_id"].isin(test_ids), "label"] = "incorrect"
    pilot.stage_report(flipped, paths, "synthetic", encode=_fake_encode)
    assert json.loads(paths.thresholds.read_text()) == first


def test_plan_counts_calls_before_rubrics_exist(offline_paths, big):
    paths = offline_paths
    pilot.stage_split(big, paths)
    llm = llm_cache.CachedLLM(paths.data / "cache", model="m1", max_calls=0, call=FakeCall())
    plan = pilot.stage_plan(big, paths, llm, offline=False)
    # Batched design: rubric batches of up to 10 questions, one grading call per question, 10 check calls
    assert plan["rubric_calls"] == -(-plan["questions"] // 10)
    assert plan["grading_calls"] == plan["questions"] and plan["check_calls"] == 10
    assert plan["total_calls"] == plan["rubric_calls"] + plan["questions"] + 10


def test_cli_refuses_unsafe_combinations():
    with pytest.raises(SystemExit):
        pilot.main(["all", "--dataset", "engsaf", "--data", "x"])
    with pytest.raises(SystemExit):
        pilot.main(["plan", "--dataset", "mohler", "--data", "x", "--offline"])


def test_real_runs_default_to_no_api_calls(offline_paths, big, monkeypatch):
    """Without --max-calls, a rubric run must stop before calling the API."""
    monkeypatch.setattr(datasets, "LOADERS", {**datasets.LOADERS, "synthetic": lambda path=None: big})
    monkeypatch.setattr(pilot, "CACHE_DIR", offline_paths.data / "cache")
    called = FakeCall()
    monkeypatch.setattr(llm_cache.llm_client, "generate_json", called)
    monkeypatch.setattr(llm_cache.llm_client, "model_name", lambda: "m1")
    assert pilot.main(["split", "--dataset", "synthetic"]) == 0
    assert pilot.main(["rubrics", "--dataset", "synthetic"]) == 2
    assert called.calls == []


# ---------------------------------------------------------------- numeric gold scores (Mohler-shaped)
def _with_scores(df, seed=0):
    """Adds 0-5 gold and grader scores that follow the labels, like Mohler."""
    rng = np.random.default_rng(seed)
    base = df["label"].map({"incorrect": 1.0, "partial": 3.0, "correct": 4.75}).to_numpy()
    g1 = np.clip(base + rng.normal(0, 0.5, len(df)), 0, 5)
    g2 = np.clip(base + rng.normal(0, 0.5, len(df)), 0, 5)
    return df.assign(score=(g1 + g2) / 2, score_grader_1=g1, score_grader_2=g2)


def test_cosine_only_report_with_primary_scores_and_ceiling(offline_paths, big):
    paths = offline_paths
    df = datasets.validate(_with_scores(big))
    pilot.stage_split(df, paths)
    table = pilot.stage_report(df, paths, "synthetic", encode=_fake_encode)  # no grades: cosine methods only
    assert "rubric_tuned" not in set(table["method"])
    assert table["method"].iloc[-1] == "human ceiling: grader 1 vs grader 2"
    primary = pd.read_csv(paths.test / "primary_scores.csv")
    assert list(primary["method"]) == ["v1", "b0", "human ceiling: grader 1 vs grader 2"]
    assert (primary["n"] == 200).all()
    assert primary.loc[2, "pearson"] > 0.8  # graders agree in this synthetic data
    md = (paths.test / "metrics.md").read_text(encoding="utf-8")
    assert md.index("primary") < md.index("secondary")
    maps = json.loads(paths.thresholds.read_text())["linear_maps_to_gold_scale"]
    assert set(maps) == {"v1", "b0"}


def test_linear_map_uses_dev_only(offline_paths, big):
    paths = offline_paths
    df = datasets.validate(_with_scores(big))
    pilot.stage_split(df, paths)
    pilot.stage_report(df, paths, "synthetic", encode=_fake_encode)
    first = json.loads(paths.thresholds.read_text())
    test_ids = set(sampling.load_split(paths.split).query("split == 'test'")["item_id"])
    changed = df.copy()
    changed.loc[changed["item_id"].isin(test_ids), "score"] = 0.0
    pilot.stage_report(changed, paths, "synthetic", encode=_fake_encode)
    assert json.loads(paths.thresholds.read_text()) == first


def test_linear_map_fits_and_clips():
    m = pilot.linear_map([0, 50, 100], [1, 3, 5], 0, 5)
    assert m["slope"] == pytest.approx(0.04) and m["intercept"] == pytest.approx(1.0)
    assert list(pilot.apply_map([0, 100, 200, -100], m)) == pytest.approx([1, 5, 5, 0])
    flat = pilot.linear_map([7, 7], [2, 4], 0, 5)
    assert list(pilot.apply_map([0, 100], flat)) == [3, 3]


def test_daily_quota_stops_at_once(tmp_path):
    error = ApiError(429)
    error.args = ("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier. "
                  "Please retry in 7h12m3.5s.",)
    slept = []
    call = FakeCall([error])
    llm = llm_cache.CachedLLM(tmp_path, model="m1", call=call, max_calls=5, sleep=slept.append)
    with pytest.raises(llm_cache.QuotaStop, match=r"daily quota used up \(the API says: retry in 7h12m3\.5s\)"):
        llm.generate_json("p")
    assert slept == [] and len(call.calls) == 1


def test_cache_records_token_usage(tmp_path, monkeypatch):
    usage = {"model": "m1", "prompt_token_count": 100, "candidates_token_count": 20,
             "thoughts_token_count": 5, "total_token_count": 125}

    def fake_generate_json(prompt, *, schema=None, system=None, temperature=None, model=None):
        llm_cache.llm_client.last_usage = usage
        return {"ok": prompt}

    monkeypatch.setattr(llm_cache.llm_client, "generate_json", fake_generate_json)
    llm = llm_cache.CachedLLM(tmp_path, model="m1", max_calls=5)
    llm.generate_json("a")
    llm.generate_json("b")
    llm.generate_json("a")  # cached: not counted again
    assert llm.tokens() == {"prompt_token_count": 200, "candidates_token_count": 40,
                            "thoughts_token_count": 10, "total_token_count": 250}
    records = [json.loads(f.read_text(encoding="utf-8")) for f in (tmp_path / "m1").glob("*.json")]
    assert all(r["usage"] == usage for r in records)
