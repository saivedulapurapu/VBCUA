from pydantic import BaseModel, computed_field
from datetime import datetime
from typing import Optional
from backend.app.schemas.concept import ConceptResponse
from backend.app.services.audio_analysis import recording_warning as _recording_warning

class EvaluationBase(BaseModel):
    concept_id: int

class EvaluationCreate(EvaluationBase):
    pass

class EvaluationResponse(BaseModel):
    id: int
    concept_id: int
    transcription: str
    asr_model: Optional[str] = None
    asr_confidence: Optional[float] = None
    semantic_score: float
    filler_words_count: int
    filler_words_ratio: float
    pause_ratio: float
    rms_energy: float
    speech_duration: float
    final_score: float
    understanding_grade: str
    feedback: str
    audio_path: Optional[str] = None
    created_at: datetime
    concept: Optional[ConceptResponse] = None

    @computed_field
    @property
    def recording_warning(self) -> Optional[str]:
        """Microphone warning derived from the raw rms_energy (not a score)."""
        return _recording_warning(self.rms_energy)

    class Config:
        from_attributes = True
