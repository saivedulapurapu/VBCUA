"""Delivery measures: stored for later analysis, not scored."""

import numpy as np
import pytest
import soundfile as sf

from backend.app.services import evaluation as scoring
from backend.app.services.audio_analysis import (
    LONG_PAUSE_S, analyze_audio_features, compute_delivery_measures, count_words,
)

SR = 16000


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


def test_measures_do_not_change_fluency_score():
    # The measures are not inputs to scoring: fluency depends on the pause ratio only
    assert scoring.compute_fluency_score(0.0, 0.2)["fluency_score"] == 100.0
