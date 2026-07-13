import os
import logging
from backend.app.config import settings

logger = logging.getLogger(__name__)

# Lazy load whisper to avoid slow startup times
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        logger.info(f"Loading local Whisper model: {settings.WHISPER_MODEL}...")
        _whisper_model = whisper.load_model(settings.WHISPER_MODEL)
    return _whisper_model

def transcribe_audio_whisper(audio_path: str) -> str:
    """
    Transcribes audio using a local Whisper model.
    """
    model = get_whisper_model()
    result = model.transcribe(audio_path)
    return result.get("text", "").strip()

def transcribe_audio_gemini(audio_path: str, api_key: str) -> str:
    """
    Transcribes audio using the Google Gemini API.
    """
    import google.generativeai as genai
    
    logger.info("Transcribing audio using Google Gemini API...")
    genai.configure(api_key=api_key)
    
    # Upload file to Gemini API
    audio_file = genai.upload_file(path=audio_path)
    
    # Generate content using the file reference
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = "Please transcribe this audio recording exactly. Provide only the plain transcription text without any commentary or headers."
    
    response = model.generate_content([audio_file, prompt])
    
    # Clean up the file on the Gemini server
    try:
        audio_file.delete()
    except Exception as e:
        logger.warning(f"Failed to delete uploaded audio file from Gemini: {e}")
        
    return response.text.strip()

def transcribe_audio(audio_path: str) -> str:
    """
    Orchestrated transcription service that tries local Whisper first,
    and falls back to Gemini API if the local Whisper module fails or is unavailable.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Case 1: If Gemini API key is configured, Gemini is much faster on CPU machines
    if settings.GEMINI_API_KEY:
        try:
            return transcribe_audio_gemini(audio_path, settings.GEMINI_API_KEY)
        except Exception as e:
            logger.error(f"Gemini transcription failed, falling back to local Whisper: {e}")

    # Case 2: Local Whisper
    try:
        return transcribe_audio_whisper(audio_path)
    except Exception as e:
        logger.error(f"Local Whisper transcription failed: {e}")
        raise RuntimeError(f"Transcription failed: {e}")
