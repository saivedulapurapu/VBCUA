import os
import re
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Any, Optional, Tuple

# Mean RMS over the whole clip below this suggests a too-quiet recording.
# Uncalibrated; used only for a microphone warning, never for scoring.
LOW_RMS_THRESHOLD = 0.005

def recording_warning(rms_energy: Optional[float]) -> Optional[str]:
    """
    Returns a microphone warning for a too-quiet recording, else None.
    """
    if rms_energy is not None and rms_energy < LOW_RMS_THRESHOLD:
        return "Recording is very quiet — check your microphone."
    return None

def compute_pause_ratios(non_silent_intervals: Any, total_samples: int) -> Tuple[float, float]:
    """
    Returns (pause_ratio_b0, pause_ratio_v1) from librosa.effects.split intervals.
    b0: silent share of the whole clip (leading/trailing silence included).
    v1: silent share between the start of the first and end of the last speech.
    With no speech detected both are 1.0 (all silence).
    """
    if total_samples <= 0:
        return 0.0, 0.0
    if len(non_silent_intervals) == 0:
        return 1.0, 1.0

    non_silent_samples = sum(int(end - start) for start, end in non_silent_intervals)
    pause_ratio_b0 = float((total_samples - non_silent_samples) / total_samples)

    speech_span = int(non_silent_intervals[-1][1] - non_silent_intervals[0][0])
    pause_ratio_v1 = float((speech_span - non_silent_samples) / speech_span) if speech_span > 0 else 0.0
    return pause_ratio_b0, pause_ratio_v1

def analyze_audio_features(audio_path: str, transcript: str) -> Dict[str, Any]:
    """
    Analyzes the audio file using Librosa to compute pause ratio and RMS energy,
    and analyzes the transcript for filler word frequencies.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at {audio_path}")

    # 1. Load Audio
    # Use sr=None to keep the native sampling rate
    y, sr = librosa.load(audio_path, sr=None)
    
    # Calculate duration
    duration = float(librosa.get_duration(y=y, sr=sr))
    if duration == 0:
        return {
            "duration": 0.0,
            "rms_energy": 0.0,
            "pause_ratio": 0.0,
            "pause_ratio_v1": 0.0,
            "filler_words_count": 0,
            "filler_words_ratio": 0.0,
            "filler_words_list": []
        }

    # 2. RMS Energy (speaking volume/intensity)
    rms = librosa.feature.rms(y=y)
    mean_rms = float(np.mean(rms))

    # 3. Pause Ratio Analysis
    # librosa.effects.split returns intervals of non-silent audio
    # top_db=35 means any frame below 35dB relative to max is silence
    non_silent_intervals = librosa.effects.split(y, top_db=35)
    pause_ratio, pause_ratio_v1 = compute_pause_ratios(non_silent_intervals, len(y))

    # 4. Filler Words Analysis
    # Clean the transcript and find occurrences of filler words
    fillers = ["um", "uh", "like", "ah", "you know", "basically", "actually"]
    filler_counts = {word: 0 for word in fillers}
    
    # Tokenize transcript (remove punctuation and lowercase)
    clean_transcript = re.sub(r'[^\w\s\']', ' ', transcript.lower())
    words = clean_transcript.split()
    total_words = len(words)

    # Simple word scanning
    for word in words:
        if word in filler_counts:
            filler_counts[word] += 1
            
    # Check for "you know" phrase
    you_know_count = len(re.findall(r'\byou\s+know\b', clean_transcript))
    if "you know" in filler_counts:
        filler_counts["you know"] = you_know_count
        # Adjust individual "you" and "know" counts if they were tracked (not in our fillers list, but good practice)

    total_fillers = sum(filler_counts.values())
    
    # Filler words ratio per 100 words
    filler_ratio = float((total_fillers / total_words) * 100) if total_words > 0 else 0.0

    return {
        "duration": duration,
        "rms_energy": mean_rms,
        "pause_ratio": pause_ratio,          # B0: whole clip
        "pause_ratio_v1": pause_ratio_v1,    # v1: first to last speech only
        "filler_words_count": total_fillers,
        "filler_words_ratio": filler_ratio,
        "filler_words_list": [f"{k}: {v}" for k, v in filler_counts.items() if v > 0]
    }
