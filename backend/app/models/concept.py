from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from datetime import datetime
from backend.app.database import Base

class ReferenceConcept(Base):
    __tablename__ = "reference_concepts"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(100), unique=True, nullable=False, index=True)
    keywords = Column(JSON, nullable=False)  # List of essential terms (e.g. ["supervised", "neural networks"])
    description = Column(Text, nullable=False)  # Detailed reference text
    # Where the concept came from: "library" (curated), "custom" (added by a user)
    # or "ai-generated" (reference written by Gemini for a trending or typed-in topic). NULL = older row.
    source = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class TrendingTopics(Base):
    """Topics suggested for one day, cached so the grounded search runs at most once per day."""
    __tablename__ = "trending_topics"

    date = Column(String(10), primary_key=True)          # YYYY-MM-DD
    origin = Column(String(20), nullable=False)          # "google-search"
    topics = Column(JSON, nullable=False)                # [{"title", "category", "why"}]
    sources = Column(JSON, nullable=True)                # [{"title", "uri"}]
    search_entry_point = Column(Text, nullable=True)     # Google's required search-suggestions HTML
    model = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
