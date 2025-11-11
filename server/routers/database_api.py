"""
Database Management Router
Handles export/import operations
"""
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
from datetime import datetime
import shutil
import json
import config
from database import vector_db, metadata_db

router = APIRouter()

@router.get("/export/vectors")
async def export_vectors():
    """Export vector database"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vectors_{timestamp}.json"
    filepath = config.EXPORTS_DIR / filename

    vector_db.export(filepath)

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="application/json"
    )

@router.post("/import/vectors")
async def import_vectors(file: UploadFile = File(...)):
    """Import vector database"""
    filepath = config.EXPORTS_DIR / file.filename

    with open(filepath, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    vector_db.import_from(filepath)

    return {"status": "success", "vectors_imported": len(vector_db.vectors)}

@router.get("/export/full-state")
async def export_full_state():
    """Export full state (vectors + metadata)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"full_state_{timestamp}.json"
    filepath = config.EXPORTS_DIR / filename

    # Prepare full state data
    full_state = {
        "vectors": vector_db.vectors,
        "metadata": {
            "people": {},
            "unknown_counter": metadata_db.unknown_counter
        }
    }

    # Convert metadata to JSON-serializable format
    for person_id, person in metadata_db.people.items():
        person_dict = person.dict()
        person_dict['created_at'] = person.created_at.isoformat()
        person_dict['last_seen'] = person.last_seen.isoformat()
        person_dict['entered_at'] = person.entered_at.isoformat() if person.entered_at else None
        person_dict['last_exit'] = person.last_exit.isoformat() if person.last_exit else None
        full_state["metadata"]["people"][person_id] = person_dict

    # Save to file
    with open(filepath, 'w') as f:
        json.dump(full_state, f, indent=2)

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="application/json"
    )

@router.post("/import/full-state")
async def import_full_state(file: UploadFile = File(...)):
    """Import full state (vectors + metadata)"""
    filepath = config.EXPORTS_DIR / file.filename

    with open(filepath, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    # Load full state
    with open(filepath, 'r') as f:
        full_state = json.load(f)

    # Import vectors
    if "vectors" in full_state:
        vector_db.vectors = full_state["vectors"]
        vector_db.save()

    # Import metadata
    if "metadata" in full_state:
        metadata_data = full_state["metadata"]
        metadata_db.people = {}
        
        for person_id, person_data in metadata_data.get("people", {}).items():
            # Convert string timestamps back to datetime
            person_data['created_at'] = datetime.fromisoformat(person_data['created_at'])
            person_data['last_seen'] = datetime.fromisoformat(person_data['last_seen'])
            if person_data.get('entered_at'):
                person_data['entered_at'] = datetime.fromisoformat(person_data['entered_at'])
            else:
                person_data['entered_at'] = None
            if person_data.get('last_exit'):
                person_data['last_exit'] = datetime.fromisoformat(person_data['last_exit'])
            else:
                person_data['last_exit'] = None
            person_data.setdefault('last_similarity', None)
            person_data.setdefault('visit_count', 0)
            person_data.pop('person_id', None)
            
            from models import Person
            metadata_db.people[person_id] = Person(**person_data, person_id=person_id)
        
        metadata_db.unknown_counter = metadata_data.get("unknown_counter", 0)
        metadata_db.save()

    return {
        "status": "success",
        "vectors_imported": len(vector_db.vectors),
        "people_imported": len(metadata_db.people)
    }
