"""
Synthetic speech for RQ2 (ASR robustness): Piper text-to-speech and cafeteria-noise mixing.

Voices (Piper, offline; models in research/data/tts/voices/, git-ignored):
    indian   en_US-l2arctic-medium, one Hindi-L1 speaker of L2-ARCTIC (chosen by listening)
    british  en_GB-alba-medium
    us       en_US-ryan-medium
Conditions: clean, light (DEMAND cafeteria noise at 20 dB SNR), heavy (5 dB SNR).

Piper samples noise inside the model, so the same text can sound slightly different on each
run: every clip is synthesised once, saved as 16 kHz mono WAV and reused (with its SHA-256 in
the manifest). Noisy versions are made from the saved clip on the fly, deterministically
(seeded by clip and condition), and are not stored.

    python -m research.asr.audio listen      # listening-check clips for the four Hindi-L1 speakers
    python -m research.asr.audio synth       # clean clips: 200 test answers x 3 voices (resumable)
"""
import argparse
import hashlib
import sys
import wave
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
TTS_DIR = ROOT / "research" / "data" / "tts"
VOICE_DIR = TTS_DIR / "voices"
NOISE_FILE = ROOT / "research" / "data" / "noise" / "DEMAND_PCAFETER_16k_ch01.wav"
SAMPLE_RATE = 16_000  # what Whisper uses

# L2-ARCTIC speakers whose first language is Hindi, with their ids in the Piper model
HINDI_L1_SPEAKERS = {"SVBI": 2, "TNI": 9, "ASI": 10, "RRBI": 19}  # SVBI, TNI female; ASI, RRBI male
# Chosen by ear from the four Hindi-L1 candidates (listening check, 5 dev answers each),
# before any transcription was run
INDIAN_SPEAKER: Optional[str] = "TNI"

VOICES = {
    "indian": {"model": "en_US-l2arctic-medium", "speaker": None},  # speaker = HINDI_L1_SPEAKERS[INDIAN_SPEAKER]
    "british": {"model": "en_GB-alba-medium", "speaker": None},
    "us": {"model": "en_US-ryan-medium", "speaker": None},
}
CONDITIONS = {"clean": None, "light": 20.0, "heavy": 5.0}  # SNR in dB
NOISE_SEED = 42


# ---------------------------------------------------------------- WAV helpers
def read_wav(path: Path) -> np.ndarray:
    """Mono 16-bit PCM WAV -> float32 in [-1, 1] at its own rate (checked to be 16 kHz)."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono 16-bit PCM")
        if w.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {w.getframerate()}")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return data.astype(np.float32) / 32768.0


def write_wav(path: Path, audio: np.ndarray) -> str:
    """Writes 16 kHz mono 16-bit PCM; returns the file's SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- text-to-speech
_voices: Dict[str, object] = {}


def _load(model: str):
    if model not in _voices:
        from piper import PiperVoice
        _voices[model] = PiperVoice.load(VOICE_DIR / f"{model}.onnx")
    return _voices[model]


def speaker_id(voice: str, speaker: Optional[str] = None) -> Optional[int]:
    if voice != "indian":
        return None
    name = speaker or INDIAN_SPEAKER
    if name is None:
        raise ValueError("Choose the Indian speaker first (INDIAN_SPEAKER) after the listening check")
    return HINDI_L1_SPEAKERS[name]


def synthesize(text: str, voice: str, speaker: Optional[str] = None) -> np.ndarray:
    """Speech for text in one of VOICES, as float32 at 16 kHz."""
    from piper import SynthesisConfig

    piper_voice = _load(VOICES[voice]["model"])
    config = SynthesisConfig(speaker_id=speaker_id(voice, speaker))
    chunks = list(piper_voice.synthesize(text, syn_config=config))
    if not chunks:
        raise ValueError(f"No audio for text {text[:40]!r}")
    rate = chunks[0].sample_rate
    audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
    return to_16k(audio, rate)


def to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == SAMPLE_RATE:
        return audio.astype(np.float32)
    g = np.gcd(rate, SAMPLE_RATE)
    return resample_poly(audio, SAMPLE_RATE // g, rate // g).astype(np.float32)


# ---------------------------------------------------------------- noise
_noise: Optional[np.ndarray] = None


def noise_track() -> np.ndarray:
    global _noise
    if _noise is None:
        _noise = read_wav(NOISE_FILE)
    return _noise


def clip_seed(clip_id: str, condition: str) -> int:
    digest = hashlib.sha256(f"{NOISE_SEED}:{clip_id}:{condition}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def mix(clean: np.ndarray, noise: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """
    Adds a noise segment (seeded start, looped if the clip is longer) scaled so that
    10 log10(P_speech / P_noise) = snr_db. Peak-normalised only if the mix would clip.
    """
    rng = np.random.default_rng(seed)
    if len(noise) >= len(clean):
        start = int(rng.integers(0, len(noise) - len(clean) + 1))
        segment = noise[start:start + len(clean)]
    else:
        segment = np.resize(noise, len(clean))
    p_speech = float(np.mean(clean.astype(np.float64) ** 2))
    p_noise = float(np.mean(segment.astype(np.float64) ** 2))
    if p_speech == 0 or p_noise == 0:
        return clean.copy()
    scaled = segment * np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    mixed = clean + scaled
    peak = float(np.max(np.abs(mixed)))
    return (mixed / peak * 0.99 if peak > 1.0 else mixed).astype(np.float32)


def condition_audio(clean: np.ndarray, clip_id: str, condition: str) -> np.ndarray:
    snr = CONDITIONS[condition]
    return clean if snr is None else mix(clean, noise_track(), snr, clip_seed(clip_id, condition))


# ---------------------------------------------------------------- listening check
LISTEN_N = 5
LISTEN_SEED = 7


def listening_texts(n: int = LISTEN_N) -> List[Dict[str, str]]:
    """n dev answers (never test answers), drawn with a fixed seed, for the listening check."""
    from research.asag import datasets, pilot

    df = datasets.load("mohler", ROOT / "research" / "data" / "mohler")
    dev = pilot._sample(df, pilot.Paths("mohler")).query("split == 'dev'").sort_values("item_id")
    picked = dev.sample(n=n, random_state=LISTEN_SEED).sort_values("item_id")
    return [{"item_id": r.item_id, "text": r.answer} for r in picked.itertuples()]


def make_listening_clips(out_dir: Path = TTS_DIR / "listening_check") -> Path:
    texts = listening_texts()
    lines = ["# Listening check: Hindi-L1 speakers of L2-ARCTIC (Piper en_US-l2arctic-medium)", "",
             "Same 5 dev answers for every speaker. Pick the speaker who sounds most clearly Indian-English.",
             "British and US clips of answer 1 are included for comparison.", ""]
    for i, t in enumerate(texts, 1):
        lines.append(f"{i}. ({t['item_id']}) {t['text']}")
    for name in HINDI_L1_SPEAKERS:
        for i, t in enumerate(texts, 1):
            write_wav(out_dir / f"{name}_{i}.wav", synthesize(t["text"], "indian", speaker=name))
    for voice in ("british", "us"):
        write_wav(out_dir / f"reference_{voice}_1.wav", synthesize(texts[0]["text"], voice))
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir


# ---------------------------------------------------------------- clean clips for the test answers
CLEAN_DIR = TTS_DIR / "clean"
MANIFEST = TTS_DIR / "manifest.csv"


def clip_id(item_id: str, voice: str) -> str:
    return f"{item_id}|{voice}"


def clip_path(item_id: str, voice: str, root: Path = CLEAN_DIR) -> Path:
    return root / voice / f"{item_id}.wav"


def test_answers():
    """The 200 Mohler test answers of the fixed split (item_id, question_id, answer)."""
    from research.asag import datasets, pilot

    df = datasets.load("mohler", ROOT / "research" / "data" / "mohler")
    sample = pilot._sample(df, pilot.Paths("mohler"))
    return sample[sample["split"] == "test"].sort_values("item_id").reset_index(drop=True)


def make_clean_clips(answers, voices=tuple(VOICES), root: Path = CLEAN_DIR, manifest: Path = MANIFEST, synth=None):
    """
    Synthesises every (answer, voice) once; existing clips are kept (resumable). Writes a manifest
    with the clip's duration and SHA-256. synth(text, voice) replaces Piper in tests.
    """
    import time

    import pandas as pd

    synth = synth or synthesize
    rows = pd.read_csv(manifest, dtype=str).to_dict("records") if manifest.exists() else []
    have = {(r["item_id"], r["voice"]) for r in rows}
    started, made = time.perf_counter(), 0
    try:
        for voice in voices:
            for a in answers.itertuples():
                if (a.item_id, voice) in have and clip_path(a.item_id, voice, root).exists():
                    continue
                speech = synth(a.answer, voice)
                digest = write_wav(clip_path(a.item_id, voice, root), speech)
                rows.append({"item_id": a.item_id, "voice": voice, "piper_model": VOICES[voice]["model"],
                             "speaker": INDIAN_SPEAKER if voice == "indian" else "",
                             "seconds": f"{len(speech) / SAMPLE_RATE:.2f}", "sha256": digest})
                made += 1
    finally:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(manifest, index=False)
    return {"made": made, "total": len(rows), "seconds": time.perf_counter() - started}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic speech for RQ2.")
    parser.add_argument("command", choices=["listen", "synth"])
    args = parser.parse_args(argv)
    if args.command == "listen":
        print(f"Listening-check clips in {make_listening_clips()}")
    if args.command == "synth":
        result = make_clean_clips(test_answers())
        print(f"Clean clips: {result['made']} new, {result['total']} in the manifest, {result['seconds']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
