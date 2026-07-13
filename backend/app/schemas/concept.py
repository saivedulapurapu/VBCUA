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
    created_at: datetime

    class Config:
        from_attributes = True
