from sqlalchemy import Column, Integer, Float, String, Text, DateTime, ForeignKey, JSON
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
    asr_device = Column(String(20), nullable=True)      # e.g. "cpu-fp32" or "cuda-fp16"
    semantic_score = Column(Float, nullable=False)  # Similarity out of 100
    cosine_similarity = Column(Float, nullable=True)     # raw Sentence-BERT cosine
    keyword_coverage = Column(Float, nullable=True)      # % of concept keywords mentioned
    mentioned_keywords = Column(JSON, nullable=True)
    missing_keywords = Column(JSON, nullable=True)
    understanding_score = Column(Float, nullable=True)   # 70% similarity + 30% keyword coverage
    semantic_method = Column(String(10), nullable=True)  # "v1" / "b0" (similarity + keywords)
    
    # Speech Fluency Metrics
    filler_words_count = Column(Integer, default=0)
    filler_words_ratio = Column(Float, default=0.0)  # fillers per 100 words
    pause_ratio = Column(Float, default=0.0)         # B0: silent share of the whole clip
    pause_ratio_v1 = Column(Float, nullable=True)    # v1: silent share between first and last speech
    rms_energy = Column(Float, default=0.0)          # speaking volume/intensity metric
    speech_duration = Column(Float, default=0.0)     # duration in seconds
    # Delivery measures (stored for later analysis, not scored)
    speaking_rate_wpm = Column(Float, nullable=True)    # words per minute between first and last speech
    long_pause_count = Column(Integer, nullable=True)   # silences >= 1 s between speech
    long_pauses_per_min = Column(Float, nullable=True)
    longest_pause_s = Column(Float, nullable=True)
    
    fluency_score = Column(Float, nullable=True)
    fluency_method = Column(String(10), nullable=True)   # "v1" / "b0"

    # Combined scores
    final_score = Column(Float, nullable=False, default=0.0)
    understanding_grade = Column(String(50), nullable=False)  # "Strong", "Moderate", "Poor"
    feedback = Column(Text, nullable=False)                   # Qualitative AI feedback
    feedback_source = Column(String(20), nullable=True)       # "llm" / "rule-based"
    llm_model = Column(String(50), nullable=True)             # None when rule-based
    feedback_status = Column(String(20), nullable=True)       # pending / generating / done / failed / unavailable
    feedback_started_at = Column(DateTime, nullable=True)     # when the current Gemini request started
    feedback_seconds = Column(Float, nullable=True)           # how long the Gemini feedback call took
    audio_path = Column(String(500), nullable=True)           # Path to saved audio file
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    concept = relationship("ReferenceConcept")
