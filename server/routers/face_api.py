"""
Face Recognition API Router
Handles registration and event processing
"""
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from typing import List, Optional
import config
from database import metadata_db, vector_db
from face_recognition import face_recognizer
from services.image_service import bytes_to_image, save_image

router = APIRouter()

@router.post("/register")
async def register_person(
    name: Optional[str] = Form(None),
    rpi_id: Optional[str] = Form(None),
    images: List[UploadFile] = File(...)
):
    """Register a new person with face recognition"""
    print(f"[Registration] name={name}, rpi={rpi_id}, images={len(images)}")

    if len(images) == 0:
        raise HTTPException(status_code=400, detail="No images provided")

    # Convert to OpenCV images
    cv_images = []
    for upload_file in images:
        image_bytes = await upload_file.read()
        try:
            img = bytes_to_image(image_bytes)
            cv_images.append(img)
        except Exception as e:
            print(f"  Failed to decode image: {e}")
            continue

    if len(cv_images) == 0:
        raise HTTPException(status_code=400, detail="Failed to decode any images")

    # Generate mean embedding (CPU-intensive, but shouldn't block server)
    try:
        mean_embedding = face_recognizer.compute_mean_embedding(cv_images)
    except Exception as e:
        print(f"  Error computing embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Face recognition error: {str(e)}")

    if mean_embedding is None:
        raise HTTPException(status_code=400, detail="No faces detected in images")

    # Create person entry
    person_id = metadata_db.create_person(name=name)

    # Save sample images
    for i, img in enumerate(cv_images[:5]):
        img_path = save_image(img, person_id)
        metadata_db.add_image(person_id, img_path)

    # Store vector
    vector_db.add(person_id, mean_embedding)

    print(f"[Registration] Registered: {person_id} (labeled={name is not None})")

    return {
        "status": "success",
        "person_id": person_id,
        "name": name,
        "is_labeled": name is not None,
        "images_saved": min(5, len(cv_images)),
        "total_people": len(metadata_db.people)
    }

@router.post("/event")
async def handle_event(
    direction: str = Form(...),
    rpi_id: Optional[str] = Form(None),
    timestamp: Optional[str] = Form(None),
    images: List[UploadFile] = File(...)
):
    """Handle entry/exit event with face recognition"""
    if direction not in ["enter", "leave"]:
        raise HTTPException(status_code=400, detail="Direction must be 'enter' or 'leave'")

    if len(images) == 0:
        raise HTTPException(status_code=400, detail="No images provided")

    print(f"[Event] {direction} from {rpi_id} at {timestamp}")

    # Convert to OpenCV images
    cv_images = []
    for upload_file in images:
        image_bytes = await upload_file.read()
        try:
            img = bytes_to_image(image_bytes)
            cv_images.append(img)
        except Exception as e:
            continue

    if len(cv_images) == 0:
        raise HTTPException(status_code=400, detail="Failed to decode any images")

    # Generate mean embedding (CPU-intensive, but shouldn't block server)
    try:
        mean_embedding = face_recognizer.compute_mean_embedding(cv_images)
    except Exception as e:
        print(f"  Error computing embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Face recognition error: {str(e)}")

    if mean_embedding is None:
        raise HTTPException(status_code=400, detail="No faces detected")

    # Search for match
    match_result = vector_db.search(mean_embedding)

    if match_result is None:
        # Unknown person - auto-register
        print("[Event] Unknown person detected - auto-registering")

        person_id = metadata_db.create_person(name=None)

        # Save one sample image
        img_path = save_image(cv_images[0], person_id)
        metadata_db.add_image(person_id, img_path)

        # Store vector
        vector_db.add(person_id, mean_embedding)

        new_state = "in" if direction == "enter" else "out"
        metadata_db.update_state(person_id, new_state)

        return {
            "status": "unknown_registered",
            "person_id": person_id,
            "name": None,
            "similarity": 0.0,
            "direction": direction,
            "new_state": new_state,
            "message": "Unknown person auto-registered"
        }

    person_id, similarity = match_result
    person = metadata_db.get(person_id)

    # Update state
    new_state = "in" if direction == "enter" else "out"
    old_state = person.state if person else "unknown"
    metadata_db.update_state(person_id, new_state)

    # Save additional sample image
    if person and len(person.image_paths) < config.MAX_IMAGES_PER_PERSON:
        img_path = save_image(cv_images[0], person_id)
        metadata_db.add_image(person_id, img_path)

    print(f"[Event] {person_id}: {old_state} -> {new_state} (similarity: {similarity:.3f})")

    return {
        "status": "success",
        "person_id": person_id,
        "name": person.name if person else None,
        "similarity": round(similarity, 3),
        "direction": direction,
        "new_state": new_state,
        "old_state": old_state
    }

@router.get("/state")
async def get_state():
    """Get current room state"""
    state = {}
    for person in metadata_db.get_all():
        state[person.person_id] = person.state

    present_count = sum(1 for s in state.values() if s == "in")

    return {
        "state": state,
        "total_people": len(state),
        "people_present": present_count
    }
