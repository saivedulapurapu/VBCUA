"""
Whisper transcription of the synthetic clips, exactly as the app runs it.

Uses the app's get_whisper_model and the same options as transcribe_audio_whisper
(language="en", fp16 only on CUDA). Each job is one (answer, voice, condition, model);
noisy audio is made on the fly from the saved clean clip (see audio.condition_audio).
torch is seeded per job, so Whisper's temperature-fallback sampling is reproducible.
Transcripts are saved as they are made (research/data/asr/transcripts.csv, git-ignored),
so a run can stop and resume, and the timing run's transcripts are reused.

    python -m research.asr.transcribe time            # 20 jobs through tiny, base and small; estimate
    python -m research.asr.transcribe run --models small base tiny
    python -m research.asr.transcribe status          # transcripts done per model
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from research.asr import audio

ASR_DIR = audio.ROOT / "research" / "data" / "asr"
TRANSCRIPTS = ASR_DIR / "transcripts.csv"
MODELS = ("tiny", "base", "small")
COLUMNS = ["item_id", "voice", "condition", "model", "text", "asr_confidence", "audio_seconds", "runtime_seconds"]
TIMING_N = 20
TIMING_SEED = 11


def all_jobs(manifest: Path = audio.MANIFEST) -> pd.DataFrame:
    """Every (item_id, voice, condition) for the clips in the manifest, in a fixed order."""
    clips = pd.read_csv(manifest, dtype=str)[["item_id", "voice", "seconds"]]
    jobs = clips.merge(pd.DataFrame({"condition": list(audio.CONDITIONS)}), how="cross")
    return jobs.sort_values(["voice", "condition", "item_id"]).reset_index(drop=True)


def timing_jobs(jobs: pd.DataFrame, n: int = TIMING_N, seed: int = TIMING_SEED) -> pd.DataFrame:
    return jobs.sample(n=n, random_state=seed).sort_values(["voice", "condition", "item_id"])


def _job_seed(item_id: str, voice: str, condition: str, model: str) -> int:
    return audio.clip_seed(f"{audio.clip_id(item_id, voice)}|{model}", condition) % (2**31)


def whisper_transcribe(model_name: str, samples: np.ndarray, seed: int) -> Dict:
    """Same call as backend transcribe_audio_whisper, on an in-memory 16 kHz array."""
    import torch

    from backend.app.services.device import resolve_device
    from backend.app.services.speech_to_text import get_whisper_model, mean_avg_logprob

    device = resolve_device()
    model = get_whisper_model(model_name, device)
    torch.manual_seed(seed)
    result = model.transcribe(samples.astype(np.float32), language="en", fp16=(device == "cuda"))
    return {"text": result.get("text", "").strip(), "asr_confidence": mean_avg_logprob(result.get("segments", []))}


def load_transcripts(path: Path = TRANSCRIPTS) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"item_id": str}, keep_default_na=False) if path.exists() else pd.DataFrame(columns=COLUMNS)


def run_jobs(jobs: pd.DataFrame, models: Sequence[str], path: Path = TRANSCRIPTS,
             transcribe: Optional[Callable] = None, clean_root: Path = audio.CLEAN_DIR,
             save_every: int = 25) -> pd.DataFrame:
    """Transcribes every job with every model, skipping those already saved. Returns the new rows."""
    transcribe = transcribe or whisper_transcribe
    rows = load_transcripts(path).to_dict("records")
    done = {(r["item_id"], r["voice"], r["condition"], r["model"]) for r in rows}
    new = []
    run_started = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for model in models:
            for job in jobs.itertuples():
                key = (job.item_id, job.voice, job.condition, model)
                if key in done:
                    continue
                clean = audio.read_wav(audio.clip_path(job.item_id, job.voice, clean_root))
                samples = audio.condition_audio(clean, audio.clip_id(job.item_id, job.voice), job.condition)
                started = time.perf_counter()
                result = transcribe(model, samples, _job_seed(*key))
                row = {"item_id": job.item_id, "voice": job.voice, "condition": job.condition, "model": model,
                       "text": result["text"], "asr_confidence": result["asr_confidence"],
                       "audio_seconds": round(len(samples) / audio.SAMPLE_RATE, 2),
                       "runtime_seconds": round(time.perf_counter() - started, 3)}
                rows.append(row)
                new.append(row)
                done.add(key)
                if len(new) % save_every == 0:
                    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
                    _progress(rows, len(jobs), model, run_started)
    finally:
        pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return pd.DataFrame(new, columns=COLUMNS)


def _progress(rows: List[Dict], jobs_per_model: int, model: str, started: float) -> None:
    done = sum(1 for r in rows if r["model"] == model)
    print(f"{time.strftime('%H:%M:%S')} whisper-{model}: {done}/{jobs_per_model} "
          f"({time.perf_counter() - started:.0f} s this run)", flush=True)


def status(path: Path = TRANSCRIPTS, jobs_per_model: Optional[int] = None) -> str:
    """One line per model: transcripts done / total, and the time spent so far."""
    jobs_per_model = jobs_per_model or len(all_jobs())
    t = load_transcripts(path)
    lines = []
    for model in MODELS:
        m = t[t["model"] == model]
        spent = pd.to_numeric(m["runtime_seconds"], errors="coerce").sum() / 60
        lines.append(f"whisper-{model}: {len(m)}/{jobs_per_model} done, {spent:.0f} min of transcription so far")
    return "\n".join(lines)


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake(on: bool = True) -> bool:
    """
    Windows only: asks the system not to idle into sleep (Modern Standby) while this process runs,
    like a media player does. Does not change power settings; closing the lid still sleeps.
    Returns True if the request was made.
    """
    if sys.platform != "win32":
        return False
    import ctypes

    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED if on else ES_CONTINUOUS
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))


def estimate(timed: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    """
    Full-run estimate per model from the timed jobs: mean seconds per job x number of jobs, and
    a second estimate scaled by audio length (seconds of runtime per second of audio).
    """
    total_audio = jobs["seconds"].astype(float).sum()
    out = []
    for model, t in timed.groupby("model", sort=False):
        per_job = t["runtime_seconds"].mean()
        per_audio_second = t["runtime_seconds"].sum() / t["audio_seconds"].sum()
        out.append({"model": model, "timed_jobs": len(t), "mean_s_per_job": round(per_job, 2),
                    "median_s_per_job": round(t["runtime_seconds"].median(), 2),
                    "max_s_per_job": round(t["runtime_seconds"].max(), 2),
                    "est_minutes_by_jobs": round(per_job * len(jobs) / 60, 1),
                    "est_minutes_by_audio": round(per_audio_second * total_audio / 60, 1)})
    return pd.DataFrame(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Whisper transcription for RQ2.")
    parser.add_argument("command", choices=["time", "run", "status"])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    args = parser.parse_args(argv)
    jobs = all_jobs()
    if args.command == "time":
        sample = timing_jobs(jobs)
        timed = []
        for model in args.models:
            loaded = time.perf_counter()
            from backend.app.services.device import resolve_device
            from backend.app.services.speech_to_text import get_whisper_model
            get_whisper_model(model, resolve_device())
            print(f"whisper-{model}: loaded in {time.perf_counter() - loaded:.1f} s on {resolve_device()}")
            timed.append(run_jobs(sample, [model]))
        timed = pd.concat(timed)
        if len(timed) < len(sample) * len(args.models):
            existing = load_transcripts()
            keys = sample[["item_id", "voice", "condition"]]
            timed = existing.merge(keys, on=["item_id", "voice", "condition"])
            timed = timed[timed["model"].isin(args.models)]
        table = estimate(timed, jobs)
        print(f"Timed {len(sample)} jobs per model; full run = {len(jobs)} jobs per model "
              f"({jobs['seconds'].astype(float).sum() / 60:.0f} min of audio per model)")
        print(table.to_string(index=False))
        table.to_csv(ASR_DIR / "timing.csv", index=False)
    if args.command == "status":
        print(status(jobs_per_model=len(jobs)))
    if args.command == "run":
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} start: models {args.models}, {len(jobs)} jobs each; "
              f"keep-awake {'on' if keep_awake() else 'unavailable'}", flush=True)
        try:
            new = run_jobs(jobs, args.models)
        finally:
            keep_awake(False)
        print(f"Transcribed {len(new)} new jobs; {len(load_transcripts())} in {TRANSCRIPTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
