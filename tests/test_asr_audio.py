"""RQ2 audio: WAV round trip, resampling, SNR mixing, deterministic noise, speaker choice."""
import numpy as np
import pytest

from research.asr import audio


def tone(seconds=1.0, freq=440.0, amp=0.3):
    t = np.arange(int(audio.SAMPLE_RATE * seconds)) / audio.SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_wav_round_trip(tmp_path):
    x = tone()
    digest = audio.write_wav(tmp_path / "a.wav", x)
    y = audio.read_wav(tmp_path / "a.wav")
    assert len(digest) == 64 and np.max(np.abs(x - y)) < 1e-4


def test_resample_to_16k():
    x = np.sin(2 * np.pi * 300 * np.arange(22050) / 22050).astype(np.float32)  # 1 s at 22.05 kHz
    y = audio.to_16k(x, 22050)
    assert len(y) == 16000
    assert audio.to_16k(y, 16000) is not None and len(audio.to_16k(y, 16000)) == 16000


@pytest.mark.parametrize("snr", [20.0, 5.0])
def test_mix_hits_requested_snr(snr):
    clean = tone(2.0, amp=0.1)
    noise = np.random.default_rng(0).normal(0, 0.2, audio.SAMPLE_RATE * 10).astype(np.float32)
    mixed = audio.mix(clean, noise, snr, seed=1)
    added = mixed - clean
    measured = 10 * np.log10(np.mean(clean.astype(float) ** 2) / np.mean(added.astype(float) ** 2))
    assert measured == pytest.approx(snr, abs=0.01)


def test_mix_is_deterministic_and_seed_dependent():
    clean = tone(1.0)
    noise = np.random.default_rng(0).normal(0, 0.2, audio.SAMPLE_RATE * 10).astype(np.float32)
    a = audio.mix(clean, noise, 5.0, seed=audio.clip_seed("E01.Q01.A00|indian", "heavy"))
    b = audio.mix(clean, noise, 5.0, seed=audio.clip_seed("E01.Q01.A00|indian", "heavy"))
    c = audio.mix(clean, noise, 5.0, seed=audio.clip_seed("E01.Q01.A01|indian", "heavy"))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_mix_never_clips_and_loops_short_noise():
    clean = tone(3.0, amp=0.95)
    short_noise = np.random.default_rng(0).normal(0, 0.5, audio.SAMPLE_RATE).astype(np.float32)
    mixed = audio.mix(clean, short_noise, 0.0, seed=3)
    assert len(mixed) == len(clean) and np.max(np.abs(mixed)) <= 1.0


def test_clean_condition_is_unchanged(monkeypatch):
    monkeypatch.setattr(audio, "noise_track", lambda: (_ for _ in ()).throw(AssertionError("no noise needed")))
    x = tone()
    assert audio.condition_audio(x, "id", "clean") is x
    assert audio.CONDITIONS == {"clean": None, "light": 20.0, "heavy": 5.0}


def test_indian_voice_needs_a_chosen_speaker(monkeypatch):
    monkeypatch.setattr(audio, "INDIAN_SPEAKER", None)
    with pytest.raises(ValueError, match="listening check"):
        audio.speaker_id("indian")
    assert audio.speaker_id("indian", "ASI") == 10 and audio.speaker_id("british") is None
    assert set(audio.HINDI_L1_SPEAKERS) == {"SVBI", "TNI", "ASI", "RRBI"}
