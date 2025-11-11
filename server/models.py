# server/models.py
"""
Data Models
"""
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Person(BaseModel):
    person_id: str  # Unique ID (can be "unknown_001", "unknown_002" or named)
    name: Optional[str] = None  # Display name (None if unlabeled)
    image_paths: List[str] = []  # Paths to stored images
    created_at: datetime
    last_seen: datetime
    state: str = "out"  # "in" or "out"

class DetectionEvent(BaseModel):
    timestamp: datetime
    person_id: str
    direction: str  # "enter" or "leave"
    similarity: float
    image_count: int

class RegistrationRequest(BaseModel):
    name: Optional[str] = None  # Can be None for unlabeled

class LabelRequest(BaseModel):
    person_id: str
    name: str
