"""
RQ2 report: how speech-recognition errors change the cosine scores (no Gemini calls).

For each of the 27 conditions (3 voices x 3 noise levels x 3 Whisper models) on the 200 Mohler
test answers, compared with scoring the original typed answer:
- WER: corpus word error rate against the typed answer, after Whisper's English text normaliser.
- v1 and B0 cosine (to the instructor's reference answer, Sentence-BERT as in the app):
  Pearson with the human score_avg, RMSE after the main run's dev-fitted linear map to 0-5, and the
  shift from the typed-answer score (mean absolute and mean signed, 0-100 points).
All with 95% bootstrap CIs over answers (1000 resamples, seed 42). Conditions with missing
transcripts are reported as incomplete, not scored.

    python -m research.asr.report

Writes research/results/mohler_asr/summary.csv and summary.md (aggregates only) and
research/data/asr/scores.csv (per transcript, git-ignored).
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from research import evaluate
from research.asr import audio, transcribe

RESULTS = audio.ROOT / "research" / "results" / "mohler_asr"
THRESHOLDS = audio.ROOT / "research" / "results" / "mohler" / "thresholds.json"
SCORES = transcribe.ASR_DIR / "scores.csv"
VOICE_ORDER = list(audio.VOICES)
CONDITION_ORDER = list(audio.CONDITIONS)
MODEL_ORDER = ["tiny", "base", "small"]


# ---------------------------------------------------------------- WER
_normaliser = None


def normalise(text: str) -> List[str]:
    global _normaliser
    if _normaliser is None:
        from whisper.normalizers import EnglishTextNormalizer
        _normaliser = EnglishTextNormalizer()
    return _normaliser(text or "").split()


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Word-level Levenshtein distance (substitutions + deletions + insertions)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------- scoring
def cosine_pairs(texts: List[str], references: List[str], encode: Callable) -> np.ndarray:
    """Cosine of each text with its reference; every distinct string is encoded once."""
    unique = list(dict.fromkeys(list(texts) + list(references)))
    vectors = np.asarray(encode(unique), dtype=float)
    index = {t: i for i, t in enumerate(unique)}
    a = vectors[[index[t] for t in texts]]
    b = vectors[[index[t] for t in references]]
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.where(norms > 0, (a * b).sum(axis=1) / np.where(norms > 0, norms, 1), 0.0)


def _sbert_encode() -> Callable:
    from backend.app.services.semantic import get_sbert_model

    model = get_sbert_model()
    return lambda texts: model.encode(texts, batch_size=64, show_progress_bar=False)


def score_frame(transcripts: pd.DataFrame, answers: pd.DataFrame, encode: Callable) -> pd.DataFrame:
    """
    One row per transcript plus one 'text' row per answer (the typed answer, no ASR), with WER,
    cosine, v1/B0 scores and their shift from the typed answer's score.
    """
    from backend.app.services.semantic import similarity_from_cosine, similarity_from_cosine_b0

    base = answers[["item_id", "answer", "reference", "score"]]
    text_rows = base.assign(voice="text", condition="text", model="text", text=base["answer"])
    df = pd.concat([transcripts.merge(base, on="item_id"), text_rows], ignore_index=True)
    edits, words = [], []
    for ref, hyp in zip(df["answer"], df["text"]):
        r, h = normalise(ref), normalise(hyp)
        edits.append(edit_distance(r, h))
        words.append(len(r))
    df["wer_edits"], df["ref_words"] = edits, words
    df["cosine"] = cosine_pairs(df["text"].fillna("").tolist(), df["reference"].tolist(), encode)
    df["v1"] = [similarity_from_cosine(c) for c in df["cosine"]]
    df["b0"] = [similarity_from_cosine_b0(c) for c in df["cosine"]]
    typed = df[df["model"] == "text"].set_index("item_id")
    for m in ("v1", "b0"):
        df[f"{m}_shift"] = df[m] - df["item_id"].map(typed[m])
    return df


# ---------------------------------------------------------------- summary
def corpus_wer(edits: np.ndarray, words: np.ndarray) -> float:
    total = words.sum()
    return float(edits.sum() / total) if total else math.nan


def _with_ci(name: str, fn: Callable, n: int) -> Dict[str, float]:
    low, high, _ = evaluate.bootstrap_ci(fn, n)
    return {name: fn(np.arange(n)), f"{name}_ci_low": low, f"{name}_ci_high": high}


def summarise(df: pd.DataFrame, maps: Dict[str, Dict[str, float]], n_answers: int) -> pd.DataFrame:
    rows = []
    for (voice, condition, model), g in df.groupby(["voice", "condition", "model"], sort=False):
        g = g.sort_values("item_id")
        row = {"voice": voice, "condition": condition, "model": model, "n": len(g), "complete": len(g) == n_answers}
        if row["complete"]:
            e, w, y = g["wer_edits"].to_numpy(float), g["ref_words"].to_numpy(float), g["score"].to_numpy(float)
            row.update(_with_ci("wer", lambda i: corpus_wer(e[i], w[i]), len(g)))
            for m in ("v1", "b0"):
                s, shift = g[m].to_numpy(float), g[f"{m}_shift"].to_numpy(float)
                mapped = np.clip(s * maps[m]["slope"] + maps[m]["intercept"], maps[m]["low"], maps[m]["high"])
                row.update(_with_ci(f"{m}_pearson", lambda i: evaluate.pearson(y[i], s[i]), len(g)))
                row.update(_with_ci(f"{m}_rmse", lambda i: evaluate.rmse(y[i], mapped[i]), len(g)))
                row.update(_with_ci(f"{m}_abs_shift", lambda i: float(np.mean(np.abs(shift[i]))), len(g)))
                row[f"{m}_mean_shift"] = float(np.mean(shift))
        rows.append(row)
    table = pd.DataFrame(rows)
    order = {"voice": ["text"] + VOICE_ORDER, "condition": ["text"] + CONDITION_ORDER, "model": ["text"] + MODEL_ORDER}
    for col, values in order.items():
        table[col] = pd.Categorical(table[col], values, ordered=True)
    return table.sort_values(["voice", "condition", "model"]).reset_index(drop=True)


def _ci(r: pd.Series, name: str, digits: int = 3, scale: float = 1.0) -> str:
    if not r.get("complete", False) or pd.isna(r.get(name)):
        return "pending"
    return (f"{r[name] * scale:.{digits}f} [{r[name + '_ci_low'] * scale:.{digits}f}, "
            f"{r[name + '_ci_high'] * scale:.{digits}f}]")


def markdown(table: pd.DataFrame, n_answers: int) -> str:
    lines = [
        f"# RQ2: ASR errors and cosine scores (Mohler test answers, n = {n_answers})", "",
        "Typed answer = no speech recognition. WER against the typed answer after Whisper's English normaliser. "
        "Pearson with human score_avg; RMSE after the dev-fitted map to 0-5; |shift| = mean absolute change "
        "from the typed-answer score (0-100 points). 95% bootstrap CIs (1000 resamples, seed 42).", "",
        "| Voice | Noise | Whisper | WER % | v1 Pearson | v1 RMSE | v1 abs shift | B0 Pearson | B0 abs shift |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        lines.append(f"| {r['voice']} | {r['condition']} | {r['model']} | {_ci(r, 'wer', 1, 100)} | {_ci(r, 'v1_pearson')} "
                     f"| {_ci(r, 'v1_rmse')} | {_ci(r, 'v1_abs_shift', 1)} | {_ci(r, 'b0_pearson')} | {_ci(r, 'b0_abs_shift', 1)} |")
    incomplete = table[~table["complete"]]
    if len(incomplete):
        lines += ["", f"Incomplete conditions (not scored): {len(incomplete)}."]
    wer = table[table["complete"] & (table["model"] != "text")]
    if len(wer):
        pivot = wer.pivot_table(index=["voice", "condition"], columns="model", values="wer", observed=True) * 100
        lines += ["", "## WER % by voice, noise and Whisper model", "",
                  "| Voice | Noise | " + " | ".join(str(c) for c in pivot.columns) + " |",
                  "|---|---|" + "---|" * len(pivot.columns)]
        for (voice, condition), r in pivot.iterrows():
            lines.append(f"| {voice} | {condition} | " + " | ".join("—" if pd.isna(v) else f"{v:.1f}" for v in r) + " |")
    return "\n".join(lines) + "\n"


def run(encode: Optional[Callable] = None, transcripts_path: Path = transcribe.TRANSCRIPTS,
        answers: Optional[pd.DataFrame] = None, out_dir: Path = RESULTS, scores_path: Path = SCORES,
        thresholds_path: Path = THRESHOLDS) -> pd.DataFrame:
    answers = answers if answers is not None else audio.test_answers()
    transcripts = transcribe.load_transcripts(transcripts_path)[["item_id", "voice", "condition", "model", "text"]]
    transcripts = transcripts[transcripts["item_id"].isin(set(answers["item_id"]))]
    maps = json.loads(thresholds_path.read_text(encoding="utf-8"))["linear_maps_to_gold_scale"]
    df = score_frame(transcripts, answers, encode or _sbert_encode())
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["answer", "reference"]).to_csv(scores_path, index=False)
    table = summarise(df, maps, len(answers))
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.md").write_text(markdown(table, len(answers)), encoding="utf-8")
    return table


def main(argv: Optional[List[str]] = None) -> int:
    argparse.ArgumentParser(description="RQ2 WER and cosine report (no Gemini calls).").parse_args(argv)
    table = run()
    print(markdown(table, int(table["n"].max())))
    print(f"Written to {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
