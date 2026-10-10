import os
import logging
import threading
from typing import Any, Dict, List, Optional
from backend.app.config import settings
from backend.app.services import llm_client

logger = logging.getLogger(__name__)

# Lazy load whisper to avoid slow startup times; one cached model per size
_whisper_models: Dict[str, Any] = {}
# Startup preloading and a request may ask for the same model at once
_whisper_lock = threading.Lock()

def get_whisper_model(model_name: Optional[str] = None):
    model_name = model_name or settings.WHISPER_MODEL
    with _whisper_lock:
        if model_name not in _whisper_models:
            import whisper
            logger.info(f"Loading local Whisper model: {model_name}...")
            _whisper_models[model_name] = whisper.load_model(model_name)
        return _whisper_models[model_name]

def mean_avg_logprob(segments: List[Dict[str, Any]]) -> Optional[float]:
    """
    Mean of Whisper's per-segment avg_logprob (unweighted). None if there are no segments.
    """
    values = [s["avg_logprob"] for s in segments if s.get("avg_logprob") is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)

def transcribe_audio_whisper(audio_path: str, model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribes audio using a local Whisper model.
    Returns {"text", "asr_model", "asr_confidence"}.
    """
    model_name = model_name or settings.WHISPER_MODEL
    model = get_whisper_model(model_name)
    result = model.transcribe(audio_path, language="en", fp16=False)
    return {
        "text": result.get("text", "").strip(),
        "asr_model": f"whisper-{model_name}",
        "asr_confidence": mean_avg_logprob(result.get("segments", [])),
    }

def transcribe_audio_gemini(audio_path: str) -> str:
    """
    Transcribes audio with the configured LLM (see llm_client).
    """
    logger.info(f"Transcribing audio using {llm_client.model_name()}...")
    prompt = "Please transcribe this audio recording exactly. Provide only the plain transcription text without any commentary or headers."
    return llm_client.generate_from_audio(audio_path, prompt)

def transcribe_audio(audio_path: str) -> Dict[str, Any]:
    """
    Transcribes with local Whisper (settings.WHISPER_MODEL). Gemini is used only if
    Whisper fails and STT_GEMINI_FALLBACK is enabled, so the ASR system is known per attempt.
    Returns {"text", "asr_model", "asr_confidence"}.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        result = transcribe_audio_whisper(audio_path)
        logger.info(f"Transcribed with {result['asr_model']} (asr_confidence={result['asr_confidence']})")
        return result
    except Exception as e:
        logger.error(f"Local Whisper transcription failed: {e}")
        whisper_error = e

    if settings.STT_GEMINI_FALLBACK and llm_client.is_configured():
        try:
            text = transcribe_audio_gemini(audio_path)
            logger.info(f"Transcribed with {llm_client.model_name()} (fallback)")
            return {"text": text, "asr_model": llm_client.model_name(), "asr_confidence": None}
        except Exception as e:
            logger.error(f"Gemini fallback transcription failed: {e}")

    raise RuntimeError(f"Transcription failed: {whisper_error}")
