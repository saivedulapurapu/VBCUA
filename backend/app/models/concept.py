from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from datetime import datetime
from backend.app.database import Base

class ReferenceConcept(Base):
    __tablename__ = "reference_concepts"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(100), unique=True, nullable=False, index=True)
    keywords = Column(JSON, nullable=False)  # List of essential terms (e.g. ["supervised", "neural networks"])
    description = Column(Text, nullable=False)  # Detailed reference text
    created_at = Column(DateTime, default=datetime.utcnow)
