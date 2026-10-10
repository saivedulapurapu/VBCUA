import numpy as np
import pytest
import soundfile as sf

from backend.app.services.audio_analysis import analyze_audio_features, compute_pause_ratios

SR = 16000


def tone(seconds):
    t = np.arange(int(seconds * SR)) / SR
    return 0.3 * np.sin(2 * np.pi * 220 * t)


def silence(seconds):
    return np.zeros(int(seconds * SR))


def speech_with_gap():
    # 1 s speech, 1 s pause, 1 s speech
    return np.concatenate([tone(1.0), silence(1.0), tone(1.0)])


def features(tmp_path, name, signal):
    path = tmp_path / name
    sf.write(path, signal, SR)
    return analyze_audio_features(str(path), "some transcript")


def test_leading_and_trailing_silence_do_not_change_v1(tmp_path):
    plain = features(tmp_path, "plain.wav", speech_with_gap())
    padded = features(tmp_path, "padded.wav", np.concatenate([silence(3.0), speech_with_gap(), silence(3.0)]))

    assert padded["pause_ratio_v1"] == pytest.approx(plain["pause_ratio_v1"], abs=0.02)
    assert plain["pause_ratio_v1"] == pytest.approx(1 / 3, abs=0.05)
    # B0 counts the padding as pauses: about 7 s of 9 s silent
    assert padded["pause_ratio"] > plain["pause_ratio"] + 0.3


def test_compute_pause_ratios_from_intervals():
    # speech at [100, 200) and [300, 400) in a 1000-sample clip
    b0, v1 = compute_pause_ratios(np.array([[100, 200], [300, 400]]), 1000)
    assert b0 == pytest.approx(0.8)   # 800 of 1000 silent
    assert v1 == pytest.approx(1 / 3)  # 100 of the 300 between 100 and 400


def test_compute_pause_ratios_single_interval_has_no_pause():
    assert compute_pause_ratios(np.array([[50, 150]]), 1000)[1] == 0.0


def test_compute_pause_ratios_no_speech():
    assert compute_pause_ratios(np.empty((0, 2), dtype=int), 1000) == (1.0, 1.0)
