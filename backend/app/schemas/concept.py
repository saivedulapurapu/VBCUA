from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

class ConceptBase(BaseModel):
    title: str = Field(..., max_length=100, description="Title of the concept")
    keywords: List[str] = Field(..., description="Essential terms that should be explained")
    description: str = Field(..., description="The reference definition/explanation")

class ConceptCreate(ConceptBase):
    pass

class ConceptUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=100)
    keywords: Optional[List[str]] = None
    description: Optional[str] = None

class ConceptResponse(ConceptBase):
    id: int
    source: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ConceptGenerate(BaseModel):
    topic: str = Field(..., min_length=2, max_length=120, description="Any topic to explain")


class TrendingTopic(BaseModel):
    title: str
    category: str
    why: str = ""


class TrendingSource(BaseModel):
    title: str
    uri: str


class TrendingResponse(BaseModel):
    date: str
    origin: str  # "google-search" or "curated"
    topics: List[TrendingTopic]
    sources: List[TrendingSource] = []
    search_entry_point: Optional[str] = None  # Google's search-suggestions HTML (must be displayed)
