"""
Own spoken explanations of the six rubric concepts (a small real-speech check for RQ1/RQ2).

30 recordings: 6 concepts x {good1, good2, partial1, partial2, wrong1}, named
<concept>_<type>.m4a, in research/data/self_recordings/ (git-ignored). The researcher rates
every recording with the rubric in my_ratings.csv BEFORE the system scores anything; `score`
refuses to run until every row has a holistic rating, and freezes a copy of the ratings
(with its SHA-256) first.

    python -m research.asr.self_recordings setup        # README checklist + empty my_ratings.csv
    python -m research.asr.self_recordings convert      # m4a -> 16 kHz mono wav (FFmpeg)
    python -m research.asr.self_recordings transcribe   # Whisper small, as the app runs it
    python -m research.asr.self_recordings score --max-calls 6   # cosine + rubric grader

Rubric grading is one Gemini call per concept (its 5 transcripts batched, as in the pilot), so
6 calls in total; --max-calls defaults to 0 (no calls).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from research.asag import concept_rubrics
from research.asr import audio

SELF_DIR = audio.ROOT / "research" / "data" / "self_recordings"
RATINGS = SELF_DIR / "my_ratings.csv"
TRANSCRIPTS = SELF_DIR / "transcripts.csv"
SCORES = SELF_DIR / "scores.csv"
WHISPER_MODEL = "small"

CONCEPTS = ["normalisation", "deadlock", "tcp-vs-udp", "overfitting", "virtual-memory", "process-vs-thread"]
FILE_STEMS = {"normalisation": "normalisation", "deadlock": "deadlock", "tcp-vs-udp": "tcp_vs_udp",
              "overfitting": "overfitting", "virtual-memory": "virtual_memory", "process-vs-thread": "process_vs_thread"}
TYPES = ["good1", "good2", "partial1", "partial2", "wrong1"]
MAX_KP, MAX_M = 6, 3
# The misconception each "wrong" answer states: the one students most often hold for that concept
WRONG_MISCONCEPTION = {"normalisation": "M2", "deadlock": "M1", "tcp-vs-udp": "M1",
                       "overfitting": "M1", "virtual-memory": "M1", "process-vs-thread": "M1"}
WINGET_FFMPEG = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"


def rubrics_by_id() -> Dict[str, Dict]:
    return {r["id"]: r for r in concept_rubrics.load_all()}


def partial_plan(n_key_points: int) -> Dict[str, List[str]]:
    """
    Key points each partial answer LEAVES OUT. partial1 states K1-K3 only; partial2 states K1 and
    K4 onwards but leaves out K2-K3. Every key point is thus missing from at least one partial
    answer and present in at least one (K1, the definition, is in both).
    """
    ids = [f"K{i}" for i in range(1, n_key_points + 1)]
    return {"partial1": ids[3:], "partial2": ids[1:3]}


def file_name(concept: str, kind: str) -> str:
    return f"{FILE_STEMS[concept]}_{kind}.m4a"


# ---------------------------------------------------------------- setup: checklist and rating sheet
def checklist(rubrics: Dict[str, Dict]) -> str:
    lines = [
        "# Self-recordings checklist (30 files)", "",
        "Record **6 concepts x 5 answers**, each **60-90 s**, with a phone voice recorder in a quiet room.",
        "Save as `<concept>_<type>.m4a` in this folder, e.g. `deadlock_good1.m4a`.",
        "Speak naturally, as a student explaining to a classmate; don't read the rubric aloud word for word.", "",
        "| Type | What to say |", "|---|---|",
        "| good1, good2 | Cover **every key point** correctly, no misconceptions. Make the two different (order, examples, wording). |",
        "| partial1, partial2 | Cover only the key points **not** listed under \"leave out\" below. Stay correct; just omit those. |",
        "| wrong1 | Build the explanation around the **misconception** listed below, stated clearly as if you believe it. You may mention at most one key point. |",
        "",
        "**Rate each recording in `my_ratings.csv` before running any scoring** (marks per key point 0/1/2, "
        "misconceptions yes/no, holistic 0-4). Scoring refuses to run until every row has a holistic rating.", "",
    ]
    for concept in CONCEPTS:
        r = rubrics[concept]
        plan = partial_plan(len(r["key_points"]))
        texts = {k["id"]: k["text"] for k in r["key_points"]}
        m_id = WRONG_MISCONCEPTION[concept]
        m_text = next(m["text"] for m in r["misconceptions"] if m["id"] == m_id)
        lines += [f"## {r['concept']}", ""]
        lines += [f"- [ ] `{file_name(concept, t)}`" for t in TYPES]
        lines += ["", "Key points:"] + [f"- {k}: {t}" for k, t in texts.items()]
        for kind in ("partial1", "partial2"):
            lines += ["", f"**{kind}: leave out** " + ", ".join(plan[kind]) + " (say the others)"]
        lines += ["", f"**wrong1: state {m_id}:** {m_text}", ""]
    lines += ["Source: research/rubrics/*.json (DRAFT – NOT FACULTY REVIEWED).", ""]
    return "\n".join(lines)


def rating_sheet(rubrics: Dict[str, Dict]) -> pd.DataFrame:
    """One row per file. Key points / misconceptions a rubric does not have are 'n/a'."""
    rows = []
    for concept in CONCEPTS:
        r = rubrics[concept]
        n_kp, n_m = len(r["key_points"]), len(r["misconceptions"])
        for kind in TYPES:
            row = {"file": file_name(concept, kind), "concept": concept, "type": kind}
            row.update({f"K{i}": "" if i <= n_kp else "n/a" for i in range(1, MAX_KP + 1)})
            row.update({f"M{i}": "" if i <= n_m else "n/a" for i in range(1, MAX_M + 1)})
            row.update({"holistic_0_4": "", "notes": ""})
            rows.append(row)
    return pd.DataFrame(rows)


def setup(folder: Path = SELF_DIR) -> None:
    rubrics = rubrics_by_id()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "README.md").write_text(checklist(rubrics), encoding="utf-8")
    sheet = folder / "my_ratings.csv"
    if sheet.exists():
        print(f"{sheet} already exists; not overwriting your ratings.")
    else:
        rating_sheet(rubrics).to_csv(sheet, index=False)


# ---------------------------------------------------------------- ratings must come first
def check_ratings(path: Path = RATINGS) -> pd.DataFrame:
    """Every row needs a holistic 0-4 rating, key-point marks 0/1/2 and misconceptions yes/no."""
    if not path.exists():
        raise SystemExit(f"No ratings at {path}: run `setup` and rate every recording first.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    problems = []
    for row in df.itertuples():
        if row.holistic_0_4.strip() not in {"0", "1", "2", "3", "4"}:
            problems.append(f"{row.file}: holistic_0_4 must be 0-4")
        for i in range(1, MAX_KP + 1):
            v = getattr(row, f"K{i}").strip()
            if v != "n/a" and v not in {"0", "1", "2"}:
                problems.append(f"{row.file}: K{i} must be 0, 1 or 2")
        for i in range(1, MAX_M + 1):
            v = getattr(row, f"M{i}").strip().lower()
            if v != "n/a" and v not in {"yes", "no"}:
                problems.append(f"{row.file}: M{i} must be yes or no")
    if problems:
        raise SystemExit("Rate every recording before scoring:\n  " + "\n  ".join(problems[:15])
                         + (f"\n  ... and {len(problems) - 15} more" if len(problems) > 15 else ""))
    return df


def freeze_ratings(path: Path = RATINGS) -> str:
    """Copies the ratings to my_ratings.frozen.csv with their SHA-256, before any scoring."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    frozen = path.with_name("my_ratings.frozen.csv")
    if frozen.exists() and hashlib.sha256(frozen.read_bytes()).hexdigest() != digest:
        raise SystemExit("my_ratings.csv changed after it was frozen for scoring; ratings must not change after scoring starts.")
    shutil.copyfile(path, frozen)
    path.with_name("my_ratings.frozen.sha256").write_text(digest + "\n", encoding="ascii")
    return digest


# ---------------------------------------------------------------- convert and transcribe
def find_ffmpeg() -> str:
    found = shutil.which("ffmpeg") or os.environ.get("FFMPEG_BINARY")
    if found:
        return found
    for candidate in sorted(WINGET_FFMPEG.glob("Gyan.FFmpeg*/*/bin/ffmpeg.exe")):
        return str(candidate)
    raise SystemExit("FFmpeg not found: put it on PATH or set FFMPEG_BINARY")


def ffmpeg_command(ffmpeg: str, source: Path, target: Path) -> List[str]:
    return [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(source),
            "-ac", "1", "-ar", str(audio.SAMPLE_RATE), "-c:a", "pcm_s16le", str(target)]


def convert(folder: Path = SELF_DIR, run: Callable = subprocess.run) -> List[Path]:
    ffmpeg = find_ffmpeg()
    made = []
    for source in sorted(folder.glob("*.m4a")):
        target = folder / "wav" / f"{source.stem}.wav"
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        run(ffmpeg_command(ffmpeg, source, target), check=True)
        made.append(target)
    return made


def expected_files() -> List[str]:
    return [file_name(c, t) for c in CONCEPTS for t in TYPES]


def transcribe_all(folder: Path = SELF_DIR, transcribe: Optional[Callable] = None) -> pd.DataFrame:
    """Whisper small (the app's model and options) on every converted recording; resumable."""
    from research.asr.transcribe import whisper_transcribe

    transcribe = transcribe or whisper_transcribe
    out = folder / "transcripts.csv"
    rows = pd.read_csv(out, dtype=str, keep_default_na=False).to_dict("records") if out.exists() else []
    done = {r["file"] for r in rows}
    for name in expected_files():
        wav = folder / "wav" / name.replace(".m4a", ".wav")
        if name in done or not wav.exists():
            continue
        samples = audio.read_wav(wav)
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
        result = transcribe(WHISPER_MODEL, samples, seed)
        rows.append({"file": name, "model": f"whisper-{WHISPER_MODEL}", "text": result["text"],
                     "asr_confidence": result["asr_confidence"], "seconds": round(len(samples) / audio.SAMPLE_RATE, 1)})
        pd.DataFrame(rows).to_csv(out, index=False)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- scoring
def reference_text(rubric: Dict) -> str:
    """Cosine reference for a concept: its key points joined (the concept rubrics have no reference answer)."""
    return " ".join(k["text"] for k in rubric["key_points"])


def cosine_scores(texts: List[str], references: List[str], encode: Optional[Callable] = None) -> np.ndarray:
    from research.asag.pilot import cosine_scores as pairwise

    return pairwise(pd.DataFrame({"answer": texts, "reference": references}), encode)


def score(folder: Path = SELF_DIR, llm=None, encode: Optional[Callable] = None) -> pd.DataFrame:
    """
    Cosine (v1 and B0, as in the app) for every transcript, and the rubric grader with one
    batched call per concept. Refuses to run until my_ratings.csv is complete; freezes it first.
    """
    from backend.app.services.semantic import similarity_from_cosine, similarity_from_cosine_b0
    from research.asag import grader

    check_ratings(folder / "my_ratings.csv")
    freeze_ratings(folder / "my_ratings.csv")
    transcripts = pd.read_csv(folder / "transcripts.csv", dtype=str, keep_default_na=False)
    rubrics = rubrics_by_id()
    transcripts["concept"] = [next(c for c in CONCEPTS if f.startswith(FILE_STEMS[c] + "_")) for f in transcripts["file"]]
    refs = [reference_text(rubrics[c]) for c in transcripts["concept"]]
    cos = cosine_scores(transcripts["text"].tolist(), refs, encode)
    transcripts["cosine"] = cos
    transcripts["v1_score"] = [similarity_from_cosine(c) for c in cos]
    transcripts["b0_score"] = [similarity_from_cosine_b0(c) for c in cos]
    rows = []
    for concept, group in transcripts.groupby("concept", sort=False):
        rubric = rubrics[concept]
        answers = dict(zip(group["file"], group["text"]))
        results = grader.grade_batch(grader.concept_question(rubric), concept, answers, rubric, llm) if llm else {}
        for row in group.itertuples():
            r = results.get(row.file)
            rows.append({"file": row.file, "concept": concept, "cosine": round(row.cosine, 4),
                         "v1_score": round(row.v1_score, 1), "b0_score": round(row.b0_score, 1),
                         "rubric_score": r["score"] if r else "",
                         "key_point_scores": json.dumps(r["key_point_scores"]) if r else "",
                         "misconceptions_present": json.dumps(r["misconceptions_present"]) if r else "",
                         "evidence": json.dumps(r["evidence"], ensure_ascii=False) if r else ""})
    table = pd.DataFrame(rows)
    table.to_csv(folder / "scores.csv", index=False)
    return table


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Self-recorded explanations of the six rubric concepts.")
    parser.add_argument("command", choices=["setup", "convert", "transcribe", "score"])
    parser.add_argument("--max-calls", type=int, default=0, help="approved Gemini calls (default 0: cosine only)")
    args = parser.parse_args(argv)
    if args.command == "setup":
        setup()
        print(f"Checklist and rating sheet in {SELF_DIR}")
    elif args.command == "convert":
        print(f"Converted {len(convert())} recordings")
    elif args.command == "transcribe":
        print(f"{len(transcribe_all())} transcripts in {SELF_DIR / 'transcripts.csv'}")
    elif args.command == "score":
        llm = None
        if args.max_calls:
            from research.asag.llm_cache import CachedLLM
            from research.asag.pilot import CACHE_DIR
            llm = CachedLLM(CACHE_DIR, max_calls=args.max_calls)
        table = score(llm=llm)
        print(f"Scored {len(table)} recordings -> {SCORES}" + ("" if llm else " (cosine only; no Gemini calls)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
