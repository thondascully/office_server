# server/database.py
"""
Database Management - File-based vector and metadata storage
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import threading
import config
from models import Person

class VectorDatabase:
    """Manages face embeddings (vectors)"""

    def __init__(self):
        self.vectors: Dict[str, List[float]] = {}
        self._lock = threading.Lock()  # Thread-safe locking
        self.load()

    def load(self):
        """Load vectors from JSON file"""
        with self._lock:
            if config.VECTOR_DB_FILE.exists():
                with open(config.VECTOR_DB_FILE, 'r') as f:
                    self.vectors = json.load(f)
                print(f"[Database] Loaded {len(self.vectors)} vectors")
            else:
                self.vectors = {}
                print("[Database] Created new vector database")

    def save(self):
        """Save vectors to JSON file (thread-safe, atomic write)"""
        with self._lock:
            # Atomic write: write to temp file, then rename (prevents corruption)
            temp_file = config.VECTOR_DB_FILE.with_suffix('.tmp')
            try:
                with open(temp_file, 'w') as f:
                    json.dump(self.vectors, f)
                # Atomic rename (works on Unix, Windows needs special handling)
                temp_file.replace(config.VECTOR_DB_FILE)
            except Exception as e:
                # Clean up temp file on error
                if temp_file.exists():
                    temp_file.unlink()
                raise
        # Saved silently - no log needed for every save

    def add(self, person_id: str, vector: np.ndarray):
        """Add or update vector for person with validation"""
        # Validate vector before storing
        vector_norm = np.linalg.norm(vector)
        if vector_norm < 1e-6:
            print(f"[Database] ERROR: Cannot store vector for {person_id} - norm too small ({vector_norm:.6f})")
            raise ValueError(f"Invalid vector for {person_id}: norm too small")
        
        # Ensure vector is normalized
        if abs(vector_norm - 1.0) > 0.01:
            print(f"[Database] WARNING: Vector for {person_id} not normalized (norm={vector_norm:.6f}), normalizing")
            vector = vector / vector_norm
        
        # Check if vector is too generic (all values very similar)
        vector_std = np.std(vector)
        if vector_std < 0.01:
            print(f"[Database] WARNING: Vector for {person_id} has very low variance ({vector_std:.6f}), might be too generic")
        
        self.vectors[person_id] = vector.tolist()
        self.save()
        print(f"[Database] Stored vector for {person_id} (norm={np.linalg.norm(vector):.6f}, std={vector_std:.6f})")

    def get(self, person_id: str) -> Optional[np.ndarray]:
        """Get vector for person"""
        if person_id in self.vectors:
            return np.array(self.vectors[person_id])
        return None

    def search(self, query_vector: np.ndarray, threshold: float = config.SIMILARITY_THRESHOLD) -> Optional[Tuple[str, float]]:
        """Find best matching person with diagnostic logging"""
        if not self.vectors:
            print("[Database] No vectors in database")
            return None

        # Validate query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm < 1e-6:
            print(f"[Database] ERROR: Query vector norm too small ({query_norm:.6f}), possible corruption")
            return None
        
        query_normalized = query_vector / query_norm
        
        # Validate normalized query
        normalized_norm = np.linalg.norm(query_normalized)
        if abs(normalized_norm - 1.0) > 0.01:
            print(f"[Database] WARNING: Query vector normalization issue (norm={normalized_norm:.6f})")
        
        best_match = None
        best_similarity = -1.0
        all_similarities = []

        for person_id, stored_vector in self.vectors.items():
            stored_array = np.array(stored_vector)
            
            # Validate stored vector
            stored_norm = np.linalg.norm(stored_array)
            if stored_norm < 1e-6:
                print(f"[Database] WARNING: Stored vector for {person_id} has norm too small ({stored_norm:.6f}), possible corruption")
                continue
            
            stored_normalized = stored_array / stored_norm
            similarity = float(np.dot(query_normalized, stored_normalized))
            all_similarities.append((person_id, similarity))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_id

        # Log all similarities for debugging
        if len(all_similarities) > 0:
            all_similarities.sort(key=lambda x: x[1], reverse=True)
            print(f"[Database] Similarity scores: {[(pid, f'{sim:.4f}') for pid, sim in all_similarities[:5]]}")
        
        # Sanity check: if similarity is suspiciously high (>0.95), it might be a problem
        if best_similarity > 0.95:
            print(f"[Database] WARNING: Very high similarity ({best_similarity:.4f}) - verify this is correct!")
            print(f"[Database] This could indicate:")
            print(f"[Database]   1. Same person (correct)")
            print(f"[Database]   2. Corrupted/generic embeddings (incorrect)")
            print(f"[Database]   3. Face detection issues (incorrect)")

        if best_similarity >= threshold:
            print(f"[Database] Match found: {best_match} (similarity: {best_similarity:.4f}, threshold: {threshold})")
            return best_match, best_similarity
        else:
            print(f"[Database] No match (best: {best_similarity:.4f} < threshold: {threshold})")
            return None

    def delete(self, person_id: str):
        """Delete person's vector"""
        if person_id in self.vectors:
            del self.vectors[person_id]
            self.save()

    def export(self, filepath: Path):
        """Export database to file"""
        with open(filepath, 'w') as f:
            json.dump(self.vectors, f, indent=2)

    def import_from(self, filepath: Path):
        """Import database from file"""
        with open(filepath, 'r') as f:
            imported = json.load(f)
        self.vectors.update(imported)
        self.save()


class MetadataDatabase:
    """Manages person metadata (names, images, state)"""

    def __init__(self):
        self.people: Dict[str, Person] = {}
        self.unknown_counter = 0
        self._lock = threading.Lock()  # Thread-safe locking
        self.load()

    def load(self):
        """Load metadata from JSON file"""
        with self._lock:
            if config.METADATA_FILE.exists():
                with open(config.METADATA_FILE, 'r') as f:
                    data = json.load(f)

                self.people = {}
                for person_id, person_data in data['people'].items():
                    # Convert string timestamps back to datetime
                    person_data['created_at'] = datetime.fromisoformat(person_data['created_at'])
                    person_data['last_seen'] = datetime.fromisoformat(person_data['last_seen'])
                    # Handle entered_at (may not exist in old data)
                    if 'entered_at' in person_data and person_data['entered_at']:
                        person_data['entered_at'] = datetime.fromisoformat(person_data['entered_at'])
                    else:
                        person_data['entered_at'] = None
                    # Handle last_similarity, visit_count, last_exit (may not exist in old data)
                    person_data.setdefault('last_similarity', None)
                    person_data.setdefault('visit_count', 0)
                    if 'last_exit' in person_data and person_data['last_exit']:
                        person_data['last_exit'] = datetime.fromisoformat(person_data['last_exit'])
                    else:
                        person_data['last_exit'] = None
                    # Remove person_id from person_data if it exists to avoid duplicate
                    person_data.pop('person_id', None)
                    self.people[person_id] = Person(**person_data, person_id=person_id)

                self.unknown_counter = data.get('unknown_counter', 0)
                print(f"[Database] Loaded {len(self.people)} people")
            else:
                self.people = {}
                self.unknown_counter = 0
                print("[Database] Created new metadata database")

    def save(self):
        """Save metadata to JSON file (thread-safe, atomic write)"""
        with self._lock:
            data = {
                'people': {},
                'unknown_counter': self.unknown_counter
            }

            for person_id, person in self.people.items():
                person_dict = person.dict()
                # Convert datetime to string for JSON
                person_dict['created_at'] = person.created_at.isoformat()
                person_dict['last_seen'] = person.last_seen.isoformat()
                person_dict['entered_at'] = person.entered_at.isoformat() if person.entered_at else None
                person_dict['last_exit'] = person.last_exit.isoformat() if person.last_exit else None
                data['people'][person_id] = person_dict

            # Atomic write: write to temp file, then rename (prevents corruption)
            temp_file = config.METADATA_FILE.with_suffix('.tmp')
            try:
                with open(temp_file, 'w') as f:
                    json.dump(data, f, indent=2)
                # Atomic rename (works on Unix, Windows needs special handling)
                temp_file.replace(config.METADATA_FILE)
            except Exception as e:
                # Clean up temp file on error
                if temp_file.exists():
                    temp_file.unlink()
                raise

    def create_person(self, name: Optional[str] = None, image_paths: List[str] = []) -> str:
        """Create new person entry"""
        if name:
            person_id = name.lower().replace(' ', '_')
        else:
            # Auto-generate ID for unlabeled person
            self.unknown_counter += 1
            person_id = f"unknown_{self.unknown_counter:03d}"

        # Ensure unique ID
        base_id = person_id
        counter = 1
        while person_id in self.people:
            person_id = f"{base_id}_{counter}"
            counter += 1

        person = Person(
            person_id=person_id,
            name=name,
            image_paths=image_paths,
            created_at=datetime.now(timezone.utc),
            last_seen=datetime.now(timezone.utc)
        )

        self.people[person_id] = person
        self.save()
        return person_id

    def get(self, person_id: str) -> Optional[Person]:
        """Get person by ID"""
        return self.people.get(person_id)

    def update_name(self, person_id: str, name: str):
        """Update person's name"""
        if person_id in self.people:
            self.people[person_id].name = name
            self.save()

    def add_image(self, person_id: str, image_path: str):
        """Add image to person's collection"""
        if person_id in self.people:
            self.people[person_id].image_paths.append(image_path)

            # Keep only last N images
            if len(self.people[person_id].image_paths) > config.MAX_IMAGES_PER_PERSON:
                # Remove oldest image file
                old_path = Path(self.people[person_id].image_paths[0])
                if old_path.exists():
                    old_path.unlink()
                self.people[person_id].image_paths = self.people[person_id].image_paths[-config.MAX_IMAGES_PER_PERSON:]

            self.save()

    def update_state(self, person_id: str, state: str, similarity: Optional[float] = None):
        """Update person's in/out state"""
        if person_id in self.people:
            old_state = self.people[person_id].state
            self.people[person_id].state = state
            self.people[person_id].last_seen = datetime.now(timezone.utc)
            
            # Store recognition confidence if provided
            if similarity is not None:
                self.people[person_id].last_similarity = similarity
            
            # Track when they entered (when state changes from out to in)
            if old_state == "out" and state == "in":
                self.people[person_id].entered_at = datetime.now(timezone.utc)
                self.people[person_id].visit_count += 1
            elif state == "out":
                self.people[person_id].entered_at = None
                self.people[person_id].last_exit = datetime.now(timezone.utc)
            
            self.save()
    
    def update_appearance(self, person_id: str, appearance: dict):
        """Update person's appearance features"""
        if person_id in self.people:
            self.people[person_id].appearance = appearance
            self.save()
            print(f"[Database] Updated appearance for {person_id}")
    
    def get_people_in_state(self, state: str) -> List[Person]:
        """Get all people in a specific state (e.g., 'in' or 'out')"""
        return [p for p in self.people.values() if p.state == state]

    def get_unlabeled(self) -> List[Person]:
        """Get all unlabeled people"""
        return [p for p in self.people.values() if p.name is None]

    def get_all(self) -> List[Person]:
        """Get all people"""
        return list(self.people.values())

    def delete(self, person_id: str):
        """Delete person"""
        if person_id in self.people:
            # Delete associated images
            for img_path in self.people[person_id].image_paths:
                path = Path(img_path)
                if path.exists():
                    path.unlink()

            del self.people[person_id]
            self.save()


vector_db = VectorDatabase()
metadata_db = MetadataDatabase()

class Event:
    event_id: str
    person_id: str
    direction: str  # "enter" or "leave"
    timestamp: datetime
    rpi_id: str
    confidence: float

class EventDatabase:
    def __init__(self):
        self.events = []
    
    def log_event(self, person_id, direction, rpi_id, confidence):
        event = Event(
            event_id=str(uuid.uuid4()),
            person_id=person_id,
            direction=direction,
            timestamp=datetime.now(),
            rpi_id=rpi_id,
            confidence=confidence
        )
        self.events.append(event)
    
    def get_events(self, person_id=None, limit=50):
        filtered = self.events
        if person_id:
            filtered = [e for e in filtered if e.person_id == person_id]
        return sorted(filtered, key=lambda e: e.timestamp, reverse=True)[:limit]