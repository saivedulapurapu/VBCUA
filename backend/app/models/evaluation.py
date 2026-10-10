from sqlalchemy import Column, Integer, Float, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.app.database import Base

class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, index=True)
    concept_id = Column(Integer, ForeignKey("reference_concepts.id"), nullable=False)
    
    # Textual evaluation
    transcription = Column(Text, nullable=False)
    asr_model = Column(String(50), nullable=True)       # e.g. "whisper-small"
    asr_confidence = Column(Float, nullable=True)       # mean Whisper avg_logprob over segments
    semantic_score = Column(Float, nullable=False)  # Similarity out of 100
    
    # Speech Fluency Metrics
    filler_words_count = Column(Integer, default=0)
    filler_words_ratio = Column(Float, default=0.0)  # fillers per 100 words
    pause_ratio = Column(Float, default=0.0)         # B0: silent share of the whole clip
    pause_ratio_v1 = Column(Float, nullable=True)    # v1: silent share between first and last speech
    rms_energy = Column(Float, default=0.0)          # speaking volume/intensity metric
    speech_duration = Column(Float, default=0.0)     # duration in seconds
    
    # Combined scores
    final_score = Column(Float, nullable=False, default=0.0)
    understanding_grade = Column(String(50), nullable=False)  # "Strong", "Moderate", "Poor"
    feedback = Column(Text, nullable=False)                   # Qualitative AI feedback
    audio_path = Column(String(500), nullable=True)           # Path to saved audio file
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    concept = relationship("ReferenceConcept")
