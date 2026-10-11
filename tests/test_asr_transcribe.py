"""RQ2 transcription: job list, resumable runs, per-job seeds, estimates (Whisper faked)."""
import numpy as np
import pandas as pd

from research.asr import audio, transcribe


def _setup(tmp_path):
    clean = tmp_path / "clean"
    rows = []
    for item in ("E01.Q01.A00", "E01.Q01.A01"):
        for voice in audio.VOICES:
            t = np.arange(audio.SAMPLE_RATE) / audio.SAMPLE_RATE
            audio.write_wav(audio.clip_path(item, voice, clean), (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32))
            rows.append({"item_id": item, "voice": voice, "seconds": "1.00"})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return clean, manifest


class FakeWhisper:
    def __init__(self):
        self.calls = []

    def __call__(self, model, samples, seed):
        self.calls.append((model, len(samples), seed))
        return {"text": f"{model} text", "asr_confidence": -0.2}


def test_jobs_cover_every_clip_and_condition(tmp_path):
    _, manifest = _setup(tmp_path)
    jobs = transcribe.all_jobs(manifest)
    assert len(jobs) == 2 * 3 * 3
    assert set(jobs["condition"]) == {"clean", "light", "heavy"}
    assert transcribe.timing_jobs(jobs, n=5).equals(transcribe.timing_jobs(jobs, n=5))


def test_run_is_resumable_and_seeded(tmp_path, monkeypatch):
    clean, manifest = _setup(tmp_path)
    noise = np.random.default_rng(0).normal(0, 0.1, audio.SAMPLE_RATE * 5).astype(np.float32)
    monkeypatch.setattr(audio, "noise_track", lambda: noise)
    jobs = transcribe.all_jobs(manifest)
    out = tmp_path / "transcripts.csv"
    fake = FakeWhisper()
    new = transcribe.run_jobs(jobs, ["tiny", "small"], path=out, transcribe=fake, clean_root=clean, save_every=4)
    assert len(new) == 36 and len(fake.calls) == 36
    saved = transcribe.load_transcripts(out)
    assert len(saved) == 36 and set(saved["model"]) == {"tiny", "small"}
    assert (saved["audio_seconds"] == 1.0).all()
    again = transcribe.run_jobs(jobs, ["tiny", "small", "base"], path=out, transcribe=fake, clean_root=clean)
    assert len(again) == 18 and set(again["model"]) == {"base"}  # only the new model
    seeds = [s for _, _, s in fake.calls]
    assert len(set(seeds)) == len(seeds)  # one seed per (clip, condition, model)


def test_estimate_scales_by_jobs_and_audio():
    timed = pd.DataFrame({"model": ["tiny"] * 2, "runtime_seconds": [1.0, 3.0], "audio_seconds": [10.0, 10.0]})
    jobs = pd.DataFrame({"seconds": ["10"] * 60})
    row = transcribe.estimate(timed, jobs).iloc[0]
    assert row["mean_s_per_job"] == 2.0 and row["est_minutes_by_jobs"] == 2.0
    assert row["est_minutes_by_audio"] == 2.0  # 0.2 s per audio second x 600 s


def test_status_counts_per_model(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame([{"item_id": "a", "voice": "us", "condition": "clean", "model": "small", "text": "x",
                   "asr_confidence": -0.1, "audio_seconds": 1.0, "runtime_seconds": 120.0}]).to_csv(path, index=False)
    lines = transcribe.status(path, jobs_per_model=1800).splitlines()
    assert lines == ["whisper-tiny: 0/1800 done, 0 min of transcription so far",
                     "whisper-base: 0/1800 done, 0 min of transcription so far",
                     "whisper-small: 1/1800 done, 2 min of transcription so far"]
