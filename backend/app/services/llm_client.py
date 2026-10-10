"""
Single entry point for all LLM calls (Google Gemini via the google-genai SDK).
Other modules call generate_text / generate_json / generate_from_audio and never
import the SDK directly, so the provider or model can be swapped in one place.
"""
import json
import logging
import time
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel, ValidationError

from backend.app.config import settings

logger = logging.getLogger(__name__)

class LLMError(Exception):
    """The LLM call failed or returned unusable output."""

class LLMNotConfigured(LLMError):
    """No API key is set."""

# Lazy client so importing this module never needs the SDK or a key
_client = None

def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client

def is_configured() -> bool:
    return bool(settings.GEMINI_API_KEY)

def model_name() -> str:
    return settings.GEMINI_MODEL

# Temporary server-side errors ("high demand", rate limits) are retried with these delays
RETRYABLE_STATUS = {429, 500, 503}
RETRY_DELAYS_S = (2.0, 5.0)
_sleep = time.sleep

def _status_code(error: Exception) -> Optional[int]:
    code = getattr(error, "code", None)
    return code if isinstance(code, int) else None

def _generate(contents: Any, config: Dict[str, Any]) -> str:
    if not is_configured():
        raise LLMNotConfigured("GEMINI_API_KEY is not set")
    attempt = 0
    while True:
        try:
            response = _get_client().models.generate_content(
                model=model_name(),
                contents=contents,
                config=config,
            )
            break
        except LLMError:
            raise
        except Exception as e:
            status = _status_code(e)
            if status in RETRYABLE_STATUS and attempt < len(RETRY_DELAYS_S):
                delay = RETRY_DELAYS_S[attempt]
                attempt += 1
                logger.warning(f"{model_name()} returned {status}; retry {attempt}/{len(RETRY_DELAYS_S)} in {delay:.0f}s")
                _sleep(delay)
                continue
            raise LLMError(f"{model_name()} call failed: {e}") from e
    text = (response.text or "").strip()
    if not text:
        raise LLMError(f"{model_name()} returned an empty response")
    return text

# Sentinel: use settings.GEMINI_THINKING_LEVEL
DEFAULT_THINKING = object()

def _base_config(system: Optional[str], temperature: Optional[float], thinking_level: Any = DEFAULT_THINKING) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if system:
        config["system_instruction"] = system
    if temperature is not None:
        config["temperature"] = temperature
    level = settings.GEMINI_THINKING_LEVEL if thinking_level is DEFAULT_THINKING else thinking_level
    if level:
        config["thinking_config"] = {"thinking_level": str(level).upper()}
    return config

def generate_text(
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    thinking_level: Any = DEFAULT_THINKING,
) -> str:
    """
    Returns the model's plain-text answer to the prompt.
    thinking_level: "minimal" / "low" / "medium" / "high", None for the model default,
    or omitted to use settings.GEMINI_THINKING_LEVEL.
    """
    return _generate(prompt, _base_config(system, temperature, thinking_level))

def generate_json(
    prompt: str,
    *,
    schema: Union[Type[BaseModel], Dict[str, Any], None] = None,
    system: Optional[str] = None,
    temperature: Optional[float] = 0.0,
    thinking_level: Any = DEFAULT_THINKING,
) -> Dict[str, Any]:
    """
    Returns the model's answer parsed as JSON.
    schema: a Pydantic model (output is validated against it) or a JSON-schema dict.
    thinking_level: as for generate_text.
    """
    config = _base_config(system, temperature, thinking_level)
    config["response_mime_type"] = "application/json"
    is_model = isinstance(schema, type) and issubclass(schema, BaseModel)
    if schema is not None:
        config["response_json_schema"] = schema.model_json_schema() if is_model else schema

    text = _generate(prompt, config)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"{model_name()} returned invalid JSON: {e}") from e
    if is_model:
        try:
            data = schema.model_validate(data).model_dump()
        except ValidationError as e:
            raise LLMError(f"{model_name()} JSON does not match {schema.__name__}: {e}") from e
    return data

def generate_from_audio(audio_path: str, prompt: str) -> str:
    """Uploads an audio file, asks the model about it, and always deletes the upload."""
    if not is_configured():
        raise LLMNotConfigured("GEMINI_API_KEY is not set")
    client = _get_client()
    try:
        uploaded = client.files.upload(file=audio_path)
    except Exception as e:
        raise LLMError(f"Audio upload failed: {e}") from e
    try:
        return _generate([uploaded, prompt], {})
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception as e:
            logger.warning(f"Failed to delete uploaded audio file from Gemini: {e}")
