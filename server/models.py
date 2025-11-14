# server/models.py
"""
Data Models
"""
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

class Person(BaseModel):
    person_id: str  # Unique ID (can be "unknown_001", "unknown_002" or named)
    name: Optional[str] = None  # Display name (None if unlabeled)
    image_paths: List[str] = []  # Paths to stored images
    created_at: datetime
    last_seen: datetime
    state: str = "out"  # "in" or "out"
    entered_at: Optional[datetime] = None  # When they entered (None if out)
    last_similarity: Optional[float] = None  # Last recognition confidence score
    visit_count: int = 0  # Number of times they've entered
    last_exit: Optional[datetime] = None  # When they last left
    appearance: Optional[Dict] = None  # Appearance features for back-detection matching
    # Appearance structure:
    # {
    #     "height_estimate": "tall/medium/short",
    #     "shirt_color": "color name",
    #     "hair_color": "color name",
    #     "hair_length": "short/medium/long",
    #     "gender": "male/female/unknown",
    #     "description": "concise description"
    # }

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
