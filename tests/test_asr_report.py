"""RQ2 report: WER, cosine shifts, complete vs incomplete conditions (fake encoder, no models)."""
import json

import numpy as np
import pandas as pd
import pytest

from research.asr import report


def test_edit_distance():
    assert report.edit_distance("a b c".split(), "a b c".split()) == 0
    assert report.edit_distance("a b c".split(), "a x c".split()) == 1
    assert report.edit_distance("a b c".split(), "a c".split()) == 1
    assert report.edit_distance("a b".split(), "a b c d".split()) == 2
    assert report.edit_distance([], "a".split()) == 1


def test_normaliser_ignores_case_and_punctuation():
    assert report.normalise("Hello, World!") == report.normalise("hello world")


def bag_encode(texts):
    vocab = sorted({w for t in texts for w in t.lower().split()})
    out = np.zeros((len(texts), len(vocab) + 1))
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, vocab.index(w)] += 1
        out[i, -1] = 0.01  # never a zero vector
    return out


@pytest.fixture
def setup(tmp_path):
    answers = pd.DataFrame({
        "item_id": [f"E01.Q01.A{i:02d}" for i in range(6)],
        "answer": ["stack is last in first out", "a queue is first in first out", "pointers hold addresses",
                   "trees have nodes and edges", "hash tables map keys to values", "arrays store elements"],
        "reference": ["a stack is last in first out"] * 6,
        "score": [5.0, 1.0, 2.0, 3.0, 2.5, 4.0],
    })
    rows = []
    for voice in ("indian", "british"):
        for condition in ("clean", "heavy"):
            for a in answers.itertuples():
                text = a.answer if condition == "clean" else " ".join(a.answer.split()[:-1])  # drop last word
                rows.append({"item_id": a.item_id, "voice": voice, "condition": condition, "model": "tiny", "text": text,
                             "asr_confidence": -0.2, "audio_seconds": 1.0, "runtime_seconds": 1.0})
    rows = rows[:-1]  # british/heavy incomplete
    transcripts = tmp_path / "transcripts.csv"
    pd.DataFrame(rows).to_csv(transcripts, index=False)
    thresholds = tmp_path / "thresholds.json"
    maps = {m: {"slope": 0.03, "intercept": 1.0, "low": 0.0, "high": 5.0} for m in ("v1", "b0")}
    thresholds.write_text(json.dumps({"linear_maps_to_gold_scale": maps}))
    return answers, transcripts, thresholds


def test_report_end_to_end(tmp_path, setup):
    answers, transcripts, thresholds = setup
    table = report.run(encode=bag_encode, transcripts_path=transcripts, answers=answers,
                       out_dir=tmp_path / "out", scores_path=tmp_path / "scores.csv", thresholds_path=thresholds)
    key = lambda v, c: table[(table["voice"] == v) & (table["condition"] == c)].iloc[0]
    text, clean, heavy, partial = key("text", "text"), key("indian", "clean"), key("indian", "heavy"), key("british", "heavy")
    assert text["wer"] == 0 and clean["wer"] == 0 and clean["v1_abs_shift"] == pytest.approx(0)
    words = sum(len(a.split()) for a in answers["answer"])
    assert heavy["wer"] == pytest.approx(6 / words)  # one deleted word per answer
    assert heavy["v1_abs_shift"] > 0 and heavy["wer_ci_low"] <= heavy["wer"] <= heavy["wer_ci_high"]
    assert not partial["complete"] and partial["n"] == 5 and pd.isna(partial.get("wer"))
    assert text["v1_pearson"] == pytest.approx(clean["v1_pearson"])
    md = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "| text | text | text |" in md and "pending" in md and "Incomplete conditions (not scored): 1." in md
    scores = pd.read_csv(tmp_path / "scores.csv")
    assert "answer" not in scores.columns and len(scores) == 23 + 6


def test_b0_shift_is_half_of_v1_for_positive_cosines(tmp_path, setup):
    answers, transcripts, thresholds = setup
    table = report.run(encode=bag_encode, transcripts_path=transcripts, answers=answers,
                       out_dir=tmp_path / "out", scores_path=tmp_path / "scores.csv", thresholds_path=thresholds)
    heavy = table[(table["voice"] == "indian") & (table["condition"] == "heavy")].iloc[0]
    assert heavy["b0_abs_shift"] == pytest.approx(heavy["v1_abs_shift"] / 2)
