"""
Face Recognition API Router
Handles registration and event processing
"""
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Request
from typing import List, Optional
import time
import asyncio
import config
from database import metadata_db, vector_db
from face_recognition import face_recognizer
from services.image_service import bytes_to_image, save_image
from services.appearance_service import appearance_service
from services.rpi_manager import rpi_manager

router = APIRouter()

@router.post("/register")
async def register_person(
    request: Request,
    name: Optional[str] = Form(None),
    rpi_id: Optional[str] = Form(None),
    images: List[UploadFile] = File(...)
):
    """Register a new person with face recognition"""
    print(f"[Registration] name={name}, rpi={rpi_id}, images={len(images)}")

    if len(images) == 0:
        raise HTTPException(status_code=400, detail="No images provided")

    # Convert to OpenCV images (limit to 5 for faster processing)
    max_images = 5
    cv_images = []
    for upload_file in images[:max_images]:  # Only process first 5
        image_bytes = await upload_file.read()
        try:
            img = bytes_to_image(image_bytes)
            cv_images.append(img)
        except Exception as e:
            print(f"  Failed to decode image: {e}")
            continue

    if len(cv_images) == 0:
        raise HTTPException(status_code=400, detail="Failed to decode any images")
    
    if len(images) > max_images:
        print(f"[Registration] Processed {len(cv_images)}/{len(images)} images (limited to {max_images} for speed)")

    # Generate mean embedding (CPU-intensive, run in thread pool to avoid blocking)
    try:
        executor = request.app.state.face_recognition_executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: face_recognizer.compute_mean_embedding(cv_images, min_embeddings=2, tripwire_zone=None)
        )
    except Exception as e:
        print(f"  Error computing embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Face recognition error: {str(e)}")

    if result is None:
        raise HTTPException(status_code=400, detail="No faces detected in images")

    mean_embedding, bbox_map = result

    # Create person entry
    person_id = metadata_db.create_person(name=name)

    # Save sample images with bounding boxes
    for i, img in enumerate(cv_images[:5]):
        bbox_data = bbox_map.get(i)  # Get (bbox, confidence) for this image index, or None if not detected
        if bbox_data is not None:
            bbox, confidence = bbox_data
            img_path = save_image(img, person_id, bbox=bbox, confidence=confidence)
        else:
            img_path = save_image(img, person_id, bbox=None)
        metadata_db.add_image(person_id, img_path)

    # Store vector
    vector_db.add(person_id, mean_embedding)

    # Extract appearance features from first image with face
    if bbox_map and len(bbox_map) > 0:
        first_index_with_face = min(bbox_map.keys())
        if first_index_with_face < len(cv_images):
            appearance_image = cv_images[first_index_with_face]
            print(f"[Registration] Extracting appearance features for {person_id}...")
            try:
                executor = request.app.state.face_recognition_executor
                loop = asyncio.get_event_loop()
                appearance = await loop.run_in_executor(
                    executor,
                    lambda: appearance_service.extract_appearance(appearance_image)
                )
                if appearance:
                    metadata_db.update_appearance(person_id, appearance)
                    print(f"[Registration] Appearance features extracted and stored for {person_id}")
            except Exception as e:
                print(f"[Registration] Failed to extract appearance: {e}")

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
    request: Request,
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

    print(f"[Event] {direction} from {rpi_id} at {timestamp}, {len(images)} images")

    # Get tripwire configuration for this RPi
    rpi_config = rpi_manager.load_config(rpi_id) if rpi_id else None
    tripwire_zone = None
    if rpi_config and 'tripwires' in rpi_config:
        tripwires = rpi_config['tripwires']
        outer_x = tripwires.get('outer_x', 800)
        inner_x = tripwires.get('inner_x', 1800)
        tripwire_zone = (outer_x, inner_x)
        print(f"[Event] Using tripwire zone: x=[{outer_x}, {inner_x}]")

    # Convert to OpenCV images (limit to 5 for faster processing)
    max_images = 5
    cv_images = []
    decode_errors = 0
    for i, upload_file in enumerate(images[:max_images]):  # Only process first 5
        try:
            image_bytes = await upload_file.read()
            if len(image_bytes) == 0:
                decode_errors += 1
                continue
            img = bytes_to_image(image_bytes)
            cv_images.append(img)
        except Exception as e:
            decode_errors += 1
            print(f"  Failed to decode image {i+1}/{min(len(images), max_images)}: {e}")
            continue

    if len(cv_images) == 0:
        error_msg = f"Failed to decode any images from {len(images)} uploaded (decode errors: {decode_errors})"
        print(f"[Event] ERROR: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    
    if len(images) > max_images:
        print(f"[Event] Processed {len(cv_images)}/{len(images)} images (limited to {max_images} for speed)")
    else:
        print(f"[Event] Successfully decoded {len(cv_images)}/{len(images)} images")

    # Generate mean embedding (CPU-intensive, run in thread pool to avoid blocking)
    start_time = time.time()
    print(f"[Event] Computing face embedding from {len(cv_images)} images...")
    try:
        executor = request.app.state.face_recognition_executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: face_recognizer.compute_mean_embedding(cv_images, min_embeddings=2, tripwire_zone=tripwire_zone)
        )
    except Exception as e:
        error_msg = f"Face recognition error: {str(e)}"
        print(f"[Event] ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

    mean_embedding = None
    bbox_map = {}
    
    # If face detection failed, try back-detection matching for "leave" events
    if result is None:
        if direction == "leave":
            print("[Event] No face detected - trying appearance-based back-detection matching")
            # Get all people currently "in" the office
            people_in = metadata_db.get_people_in_state("in")
            
            if len(people_in) > 0:
                # Build candidate appearances dict
                candidate_appearances = {}
                for person in people_in:
                    if person.appearance:
                        candidate_appearances[person.person_id] = person.appearance
                
                if len(candidate_appearances) > 0:
                    print(f"[Event] Matching against {len(candidate_appearances)} people with appearance data")
                    # Use first image for appearance matching
                    match_result_appearance = await loop.run_in_executor(
                        executor,
                        lambda: appearance_service.match_by_appearance(cv_images[0], candidate_appearances)
                    )
                    
                    if match_result_appearance:
                        person_id, appearance_score = match_result_appearance
                        print(f"[Event] Appearance match found: {person_id} (score: {appearance_score:.2f})")
                        person = metadata_db.get(person_id)
                        new_state = "out"
                        old_state = person.state if person else "unknown"
                        metadata_db.update_state(person_id, new_state, similarity=appearance_score)
                        
                        return {
                            "status": "success",
                            "person_id": person_id,
                            "name": person.name if person else None,
                            "similarity": round(appearance_score, 3),
                            "direction": direction,
                            "new_state": new_state,
                            "old_state": old_state,
                            "method": "appearance_matching"
                        }
            
            # No match found via appearance
            error_msg = "No faces detected and no appearance match found"
            print(f"[Event] ERROR: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        else:
            # For "enter" events, face is required
            error_msg = "No faces detected in any of the images"
            print(f"[Event] ERROR: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
    
    mean_embedding, bbox_map = result
    elapsed = time.time() - start_time
    print(f"[Event] Face embedding computed successfully in {elapsed:.2f}s")

    # Search for match (CPU-intensive vector comparison, run in thread pool)
    executor = request.app.state.face_recognition_executor
    loop = asyncio.get_event_loop()
    match_result = await loop.run_in_executor(
        executor,
        vector_db.search,
        mean_embedding
    )

    if match_result is None:
        # Unknown person - auto-register
        print("[Event] Unknown person detected - auto-registering")

        person_id = metadata_db.create_person(name=None)

        # Save one sample image with bounding box (use first image that had a face detected, or first image)
        bbox = None
        confidence = None
        image_to_save = cv_images[0]
        if bbox_map and len(bbox_map) > 0:
            # Use the first image index that had a face detected
            first_index_with_face = min(bbox_map.keys())
            bbox, confidence = bbox_map[first_index_with_face]
            if first_index_with_face < len(cv_images):
                image_to_save = cv_images[first_index_with_face]
        img_path = save_image(image_to_save, person_id, bbox=bbox, confidence=confidence)
        metadata_db.add_image(person_id, img_path)

        # Store vector
        vector_db.add(person_id, mean_embedding)

        # Extract appearance features if face was detected (for future back-detection)
        if direction == "enter" and bbox_map and len(bbox_map) > 0:
            first_index_with_face = min(bbox_map.keys())
            if first_index_with_face < len(cv_images):
                appearance_image = cv_images[first_index_with_face]
                print(f"[Event] Extracting appearance features for {person_id}...")
                appearance = await loop.run_in_executor(
                    executor,
                    lambda: appearance_service.extract_appearance(appearance_image)
                )
                if appearance:
                    metadata_db.update_appearance(person_id, appearance)
                    print(f"[Event] Appearance features extracted and stored for {person_id}")

        new_state = "in" if direction == "enter" else "out"
        metadata_db.update_state(person_id, new_state, similarity=0.0)

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

    # Extract/update appearance features if face was detected and entering
    if direction == "enter" and bbox_map and len(bbox_map) > 0:
        # Check if person already has appearance data, if not, extract it
        if not person.appearance:
            first_index_with_face = min(bbox_map.keys())
            if first_index_with_face < len(cv_images):
                appearance_image = cv_images[first_index_with_face]
                print(f"[Event] Extracting appearance features for {person_id}...")
                appearance = await loop.run_in_executor(
                    executor,
                    lambda: appearance_service.extract_appearance(appearance_image)
                )
                if appearance:
                    metadata_db.update_appearance(person_id, appearance)
                    print(f"[Event] Appearance features extracted and stored for {person_id}")

    # Update state
    new_state = "in" if direction == "enter" else "out"
    old_state = person.state if person else "unknown"
    metadata_db.update_state(person_id, new_state, similarity=similarity)

    # Save additional sample image with bounding box
    if person and len(person.image_paths) < config.MAX_IMAGES_PER_PERSON:
        # Find the first image that has a bounding box (face was detected)
        # If no bbox found for image 0, try to find any image with a detected face
        bbox = None
        confidence = None
        if bbox_map:
            # Try to get bbox for first image, or use first available bbox
            bbox_data = bbox_map.get(0)
            if bbox_data is not None:
                bbox, confidence = bbox_data
                img_path = save_image(cv_images[0], person_id, bbox=bbox, confidence=confidence)
            elif len(bbox_map) > 0:
                # Use the first available bounding box (face was detected in a different image)
                first_index_with_face = min(bbox_map.keys())
                bbox, confidence = bbox_map[first_index_with_face]
                # Use the corresponding image for saving
                if first_index_with_face < len(cv_images):
                    img_path = save_image(cv_images[first_index_with_face], person_id, bbox=bbox, confidence=confidence)
                else:
                    img_path = save_image(cv_images[0], person_id, bbox=bbox, confidence=confidence)
            else:
                img_path = save_image(cv_images[0], person_id, bbox=None)
        else:
            img_path = save_image(cv_images[0], person_id, bbox=None)
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
