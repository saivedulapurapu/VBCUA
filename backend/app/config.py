import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Database URL: SQLite by default; PostgreSQL is optional
    # (e.g. postgresql+psycopg2://postgres:postgres@localhost:5432/vbcua_db)
    DATABASE_URL: str = Field(
        default="sqlite:///./vbcua.db",
        validation_alias="DATABASE_URL"
    )
    
    # Gemini API Key (optional but recommended for AI features)
    GEMINI_API_KEY: str = Field(default="", validation_alias="GEMINI_API_KEY")

    # Gemini model used for all LLM calls (pinned for reproducible results)
    GEMINI_MODEL: str = Field(default="gemini-3.8-flash", validation_alias="GEMINI_MODEL")

    # Gemini model for written feedback (GEMINI_MODEL stays for the rubric grader)
    GEMINI_FEEDBACK_MODEL: str = Field(default="gemini-3.5-flash-lite", validation_alias="GEMINI_FEEDBACK_MODEL")

    # Gemini thinking effort: minimal, low, medium or high ("" = model default).
    # Low keeps feedback fast; it does not affect any score.
    GEMINI_THINKING_LEVEL: str = Field(default="low", validation_alias="GEMINI_THINKING_LEVEL")

    # Device for Whisper and Sentence-BERT: cpu (default), cuda, or auto (cuda if available).
    # Needs a CUDA build of PyTorch for cuda; falls back to cpu otherwise.
    DEVICE: str = Field(default="cpu", validation_alias="DEVICE")

    # Load Whisper and Sentence-BERT in the background at startup so the first attempt is not slowed down
    PRELOAD_MODELS: bool = Field(default=True, validation_alias="PRELOAD_MODELS")
    
    # Whisper Speech-to-Text Model Size
    WHISPER_MODEL: str = Field(default="small", validation_alias="WHISPER_MODEL")

    # Use Gemini for transcription only if local Whisper fails (off by default)
    STT_GEMINI_FALLBACK: bool = Field(default=False, validation_alias="STT_GEMINI_FALLBACK")
    
    # Directory to store uploaded audio files
    UPLOAD_DIR: str = Field(default="./uploads", validation_alias="UPLOAD_DIR")
    
    # Allow CORS origins
    CORS_ORIGINS: list[str] = ["*"]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure uploads directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
