"""Self-recordings: checklist, rating sheet, ratings-before-scoring rule, conversion, scoring (all faked)."""
import re

import numpy as np
import pandas as pd
import pytest

from research.asr import audio
from research.asr import self_recordings as sr


@pytest.fixture(scope="module")
def rubrics():
    return sr.rubrics_by_id()


def test_partial_plan_covers_every_key_point_both_ways():
    for n in (5, 6):
        plan = sr.partial_plan(n)
        ids = {f"K{i}" for i in range(1, n + 1)}
        left_out = set(plan["partial1"]) | set(plan["partial2"])
        assert left_out == ids - {"K1"}  # every key point but the definition is missing somewhere
        assert not set(plan["partial1"]) & set(plan["partial2"])  # ... and present in the other partial
    assert sr.partial_plan(5) == {"partial1": ["K4", "K5"], "partial2": ["K2", "K3"]}


def test_checklist_names_files_partials_and_misconceptions(rubrics):
    text = sr.checklist(rubrics)
    assert text.count("- [ ] `") == 30
    assert "`deadlock_good1.m4a`" in text and "`tcp_vs_udp_wrong1.m4a`" in text
    assert "60-90 s" in text and "quiet room" in text and "before running any scoring" in text
    assert "**wrong1: state M2:** Database normalisation means scaling numeric values" in text
    assert "**partial1: leave out** K4, K5, K6" in text  # 6-key-point concepts
    for concept in sr.CONCEPTS:
        assert f"## {rubrics[concept]['concept']}" in text


def test_rating_sheet_shape(rubrics):
    sheet = sr.rating_sheet(rubrics)
    assert len(sheet) == 30 and sheet["file"].is_unique
    assert list(sheet.columns) == ["file", "concept", "type", "K1", "K2", "K3", "K4", "K5", "K6",
                                   "M1", "M2", "M3", "holistic_0_4", "notes"]
    deadlock = sheet[sheet["concept"] == "deadlock"].iloc[0]  # 5 key points
    assert deadlock["K5"] == "" and deadlock["K6"] == "n/a"
    assert (sheet["holistic_0_4"] == "").all()


def test_setup_never_overwrites_ratings(tmp_path):
    sr.setup(tmp_path)
    sheet = tmp_path / "my_ratings.csv"
    assert (tmp_path / "README.md").exists() and len(pd.read_csv(sheet)) == 30
    sheet.write_text("my edits", encoding="utf-8")
    sr.setup(tmp_path)
    assert sheet.read_text(encoding="utf-8") == "my edits"


def _filled(rubrics, holistic="3"):
    sheet = sr.rating_sheet(rubrics)
    for col in [f"K{i}" for i in range(1, 7)]:
        sheet[col] = sheet[col].where(sheet[col] == "n/a", "2")
    for col in ("M1", "M2", "M3"):
        sheet[col] = sheet[col].where(sheet[col] == "n/a", "no")
    sheet["holistic_0_4"] = holistic
    return sheet


def test_scoring_refuses_until_ratings_are_complete(tmp_path, rubrics):
    path = tmp_path / "my_ratings.csv"
    sr.rating_sheet(rubrics).to_csv(path, index=False)
    with pytest.raises(SystemExit, match="Rate every recording before scoring"):
        sr.check_ratings(path)
    bad = _filled(rubrics)
    bad.loc[0, "K1"] = "3"
    bad.to_csv(path, index=False)
    with pytest.raises(SystemExit, match="K1 must be 0, 1 or 2"):
        sr.check_ratings(path)
    _filled(rubrics).to_csv(path, index=False)
    assert len(sr.check_ratings(path)) == 30


def test_frozen_ratings_cannot_change(tmp_path, rubrics):
    path = tmp_path / "my_ratings.csv"
    _filled(rubrics).to_csv(path, index=False)
    digest = sr.freeze_ratings(path)
    assert (tmp_path / "my_ratings.frozen.sha256").read_text().strip() == digest
    assert sr.freeze_ratings(path) == digest  # unchanged: fine
    _filled(rubrics, holistic="4").to_csv(path, index=False)
    with pytest.raises(SystemExit, match="changed after it was frozen"):
        sr.freeze_ratings(path)


def test_convert_calls_ffmpeg_for_new_files(tmp_path, monkeypatch):
    (tmp_path / "deadlock_good1.m4a").write_bytes(b"x")
    (tmp_path / "wav").mkdir()
    (tmp_path / "wav" / "deadlock_good2.wav").write_bytes(b"done")
    (tmp_path / "deadlock_good2.m4a").write_bytes(b"x")
    monkeypatch.setattr(sr, "find_ffmpeg", lambda: "ffmpeg.exe")
    calls = []
    made = sr.convert(tmp_path, run=lambda cmd, check: calls.append(cmd))
    assert [p.name for p in made] == ["deadlock_good1.wav"] and len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "ffmpeg.exe" and cmd[cmd.index("-ar") + 1] == "16000" and cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[-1].endswith("deadlock_good1.wav")


def test_transcribe_and_score_with_fakes(tmp_path, rubrics, monkeypatch):
    t = np.arange(audio.SAMPLE_RATE) / audio.SAMPLE_RATE
    for name in sr.expected_files():
        audio.write_wav(tmp_path / "wav" / name.replace(".m4a", ".wav"), (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32))
    seen = []

    def fake_whisper(model, samples, seed):
        seen.append(model)
        return {"text": "a deadlock is when two threads wait on each other forever", "asr_confidence": -0.3}

    transcripts = sr.transcribe_all(tmp_path, transcribe=fake_whisper)
    assert len(transcripts) == 30 and set(seen) == {"small"}
    assert len(sr.transcribe_all(tmp_path, transcribe=fake_whisper)) == 30 and len(seen) == 30  # resumable

    with pytest.raises(SystemExit):  # no ratings yet: no scoring
        sr.score(tmp_path, encode=lambda texts: np.ones((len(texts), 3)))
    _filled(rubrics).to_csv(tmp_path / "my_ratings.csv", index=False)

    from research.asag import llm_cache

    def fake_gemini(prompt, **kw):
        n_kp = len(re.findall(r"^  K\d+:", prompt, flags=re.M))
        n_m = len(re.findall(r"^  M\d+:", prompt, flags=re.M))
        return [{"answer": label, "key_points": [{"id": f"K{i}", "evidence": "two threads wait", "mark": 2} for i in range(1, n_kp + 1)],
                 "misconceptions": [{"id": f"M{i}", "evidence": "", "present": False} for i in range(1, n_m + 1)]}
                for label in re.findall(r'<answer id="(A\d+)">', prompt)]

    llm = llm_cache.CachedLLM(tmp_path / "cache", model="m1", max_calls=6, call=fake_gemini)
    table = sr.score(tmp_path, llm=llm, encode=lambda texts: np.ones((len(texts), 3)))
    assert len(table) == 30 and llm.calls == 6  # one batched call per concept
    assert set(table["rubric_score"]) == {100.0} and set(table["v1_score"]) == {100.0}
    assert (tmp_path / "my_ratings.frozen.csv").exists()


def test_cosine_only_without_llm(tmp_path, rubrics):
    t = np.arange(audio.SAMPLE_RATE) / audio.SAMPLE_RATE
    for name in sr.expected_files()[:5]:
        audio.write_wav(tmp_path / "wav" / name.replace(".m4a", ".wav"), (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32))
    sr.transcribe_all(tmp_path, transcribe=lambda m, s, seed: {"text": "redundant data", "asr_confidence": -0.1})
    _filled(rubrics).to_csv(tmp_path / "my_ratings.csv", index=False)
    table = sr.score(tmp_path, llm=None, encode=lambda texts: np.ones((len(texts), 3)))
    assert len(table) == 5 and (table["rubric_score"] == "").all()
