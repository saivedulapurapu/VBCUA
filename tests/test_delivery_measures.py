"""Delivery measures: stored for later analysis, not scored."""
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database import Base
from backend.app.models.evaluation import Evaluation
from backend.app.services import evaluation as scoring
from backend.app.services.audio_analysis import (
    LONG_PAUSE_S, analyze_audio_features, compute_delivery_measures, count_words, measure_delivery,
)

SR = 16000
ROOT = Path(__file__).resolve().parent.parent


def tone(seconds):
    t = np.arange(int(seconds * SR)) / SR
    return 0.3 * np.sin(2 * np.pi * 220 * t)


def silence(seconds):
    return np.zeros(int(seconds * SR))


def write(path, *parts):
    sf.write(path, np.concatenate(parts), SR)
    return str(path)


def test_measures_from_intervals():
    # speech 0-10 s, 12-20 s, 20.5-30 s at 1 kHz: gaps of 2.0 s and 0.5 s over a 30 s span
    intervals = np.array([[0, 10_000], [12_000, 20_000], [20_500, 30_000]])
    m = compute_delivery_measures(intervals, 1000, word_count=60)
    assert m == {"speaking_rate_wpm": 120.0, "long_pause_count": 1, "long_pauses_per_min": 2.0, "longest_pause_s": 2.0}


def test_long_pause_threshold_is_one_second():
    assert LONG_PAUSE_S == 1.0
    exactly_one = compute_delivery_measures(np.array([[0, 1000], [2000, 3000]]), 1000, 10)
    assert exactly_one["long_pause_count"] == 1


def test_single_interval_has_no_pauses():
    m = compute_delivery_measures(np.array([[0, 60_000]]), 1000, word_count=150)
    assert m == {"speaking_rate_wpm": 150.0, "long_pause_count": 0, "long_pauses_per_min": 0.0, "longest_pause_s": 0.0}


def test_no_speech_gives_none():
    assert set(compute_delivery_measures(np.empty((0, 2), dtype=int), SR, 10).values()) == {None}


def test_count_words_matches_pipeline_tokenisation():
    assert count_words("Hello, it's DBMS; a 'database' system.") == 6


def test_measured_from_real_audio(tmp_path):
    # 2 s speech, 1.5 s pause, 2 s speech, 0.4 s pause, 2 s speech (7.9 s span); padded edges are ignored
    path = write(tmp_path / "clip.wav", silence(2), tone(2), silence(1.5), tone(2), silence(0.4), tone(2), silence(2))
    features = analyze_audio_features(path, "one two three four five six seven eight nine ten")
    assert features["long_pause_count"] == 1
    # librosa detects silence in 2048-sample frames (0.128 s at 16 kHz), so edges are accurate to ~1 frame
    assert features["longest_pause_s"] == pytest.approx(1.5, abs=0.15)
    assert features["speaking_rate_wpm"] == pytest.approx(10 / (7.9 / 60), rel=0.05)
    assert measure_delivery(path, "one two three four five six seven eight nine ten") == {
        k: features[k] for k in ("speaking_rate_wpm", "long_pause_count", "long_pauses_per_min", "longest_pause_s")
    }


def test_measures_do_not_change_fluency_score():
    # The measures are not inputs to scoring: fluency depends on the pause ratio only
    assert scoring.compute_fluency_score(0.0, 0.2)["fluency_score"] == 100.0


# --- Backfill script ---

def load_backfill():
    spec = importlib.util.spec_from_file_location("backfill", ROOT / "research" / "backfill_delivery_measures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def db_factory():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


def add_attempt(factory, audio_path, **extra):
    db = factory()
    row = Evaluation(
        concept_id=1, transcription="one two three four five six", semantic_score=50, understanding_grade="Moderate Understanding",
        feedback="f", final_score=50, audio_path=audio_path, created_at=datetime(2026, 10, 10), **extra,
    )
    db.add(row)
    db.commit()
    eval_id = row.id
    db.close()
    return eval_id


def test_backfill_fills_missing_measures_only(tmp_path, db_factory):
    audio = write(tmp_path / "a.wav", tone(2), silence(1.2), tone(2))
    missing_audio = add_attempt(db_factory, str(tmp_path / "gone.wav"))
    to_fill = add_attempt(db_factory, audio)
    already = add_attempt(db_factory, audio, speaking_rate_wpm=99.0, long_pause_count=0, long_pauses_per_min=0.0, longest_pause_s=0.5)

    results = {r[0]: r[1] for r in load_backfill().backfill(db_factory)}
    assert results == {missing_audio: "audio missing", to_fill: "updated"}

    db = db_factory()
    filled = db.get(Evaluation, to_fill)
    assert filled.long_pause_count == 1 and filled.speaking_rate_wpm > 0
    assert db.get(Evaluation, already).speaking_rate_wpm == 99.0  # untouched
    db.close()


def test_backfill_dry_run_changes_nothing(tmp_path, db_factory):
    eval_id = add_attempt(db_factory, write(tmp_path / "a.wav", tone(2), silence(1.2), tone(2)))
    assert load_backfill().backfill(db_factory, dry_run=True)[0][1] == "dry run"
    db = db_factory()
    assert db.get(Evaluation, eval_id).speaking_rate_wpm is None
    db.close()
