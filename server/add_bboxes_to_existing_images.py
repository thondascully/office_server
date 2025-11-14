"""
Script to retroactively add bounding boxes to existing registered images
"""
import cv2
import sys
from pathlib import Path
from database import metadata_db
from face_recognition import face_recognizer
from services.image_service import draw_bounding_box

def add_bboxes_to_existing_images():
    """Add bounding boxes to all existing images in the database"""
    print("[BBox Script] Starting to add bounding boxes to existing images...")
    
    total_images = 0
    processed_images = 0
    failed_images = 0
    
    # Iterate through all people
    for person_id, person in metadata_db.people.items():
        print(f"\n[BBox Script] Processing person: {person_id} ({person.name or 'unnamed'})")
        print(f"  Images: {len(person.image_paths)}")
        
        for img_path in person.image_paths:
            total_images += 1
            img_file = Path(img_path)
            
            # Check if image file exists
            if not img_file.exists():
                print(f"  [WARNING] Image not found: {img_path}")
                failed_images += 1
                continue
            
            try:
                # Load image
                image = cv2.imread(str(img_file))
                if image is None:
                    print(f"  [ERROR] Failed to load image: {img_path}")
                    failed_images += 1
                    continue
                
                # Check if image already has a bounding box (simple heuristic: check if it's been modified)
                # We'll just re-detect and add the box anyway to ensure consistency
                
                # Detect face and get bounding box
                # Use get_embedding which goes through the same detection pipeline
                # that was used during registration/recognition
                result = face_recognizer.get_embedding(image)
                if result is None:
                    print(f"  [WARNING] No face detected in: {img_path}")
                    failed_images += 1
                    continue
                
                # get_embedding returns (embedding, bbox, confidence)
                embedding, bbox, confidence = result
                
                # Draw bounding box on image with confidence text
                image_with_bbox = draw_bounding_box(image.copy(), bbox, confidence=confidence)
                print(f"  [OK] Added bbox to: {img_file.name} (bbox: {bbox}, confidence: {confidence:.2%})")
                
                # Save image back (overwrite)
                cv2.imwrite(str(img_file), image_with_bbox, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                processed_images += 1
                
            except Exception as e:
                print(f"  [ERROR] Failed to process {img_path}: {e}")
                failed_images += 1
                continue
    
    print(f"\n[BBox Script] Completed!")
    print(f"  Total images: {total_images}")
    print(f"  Successfully processed: {processed_images}")
    print(f"  Failed: {failed_images}")

if __name__ == "__main__":
    try:
        add_bboxes_to_existing_images()
    except KeyboardInterrupt:
        print("\n[BBox Script] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[BBox Script] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

