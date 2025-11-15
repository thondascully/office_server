"""
Test script for appearance extraction service
Tests LLM-based appearance extraction on existing images
"""
import cv2
import sys
from pathlib import Path
from services.appearance_service import appearance_service
from database import metadata_db
import config

def test_appearance_on_image(image_path: str):
    """Test appearance extraction on a specific image file"""
    print(f"[Test] Loading image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Test] ERROR: Failed to load image: {image_path}")
        return None
    
    print(f"[Test] Image loaded: {image.shape[1]}x{image.shape[0]}")
    
    # Extract appearance
    print(f"[Test] Extracting appearance features...")
    appearance = appearance_service.extract_appearance(image)
    
    if appearance:
        print(f"\n[Test] ✅ Appearance extracted successfully!")
        print(f"[Test] Results:")
        for key, value in appearance.items():
            print(f"  {key}: {value}")
        return appearance
    else:
        print(f"\n[Test] ❌ Failed to extract appearance")
        return None

def test_appearance_on_person(person_id: str = None):
    """Test appearance extraction on a person's images"""
    if person_id is None:
        # Get first person with images
        people = list(metadata_db.people.values())
        if len(people) == 0:
            print("[Test] ERROR: No people in database")
            return
        
        person = people[0]
        person_id = person.person_id
        print(f"[Test] No person_id specified, using first person: {person_id} ({person.name or 'unnamed'})")
    else:
        person = metadata_db.get(person_id)
        if person is None:
            print(f"[Test] ERROR: Person {person_id} not found")
            return
    
    print(f"[Test] Testing on person: {person_id} ({person.name or 'unnamed'})")
    print(f"[Test] Images: {len(person.image_paths)}")
    
    if len(person.image_paths) == 0:
        print(f"[Test] ERROR: Person has no images")
        return
    
    # Test on first image
    first_image_path = person.image_paths[0]
    print(f"\n[Test] Testing on first image: {first_image_path}")
    
    return test_appearance_on_image(first_image_path)

def test_appearance_on_all_people():
    """Test appearance extraction on all people's first images"""
    print("[Test] Testing appearance extraction on all people...")
    
    results = {}
    for person_id, person in metadata_db.people.items():
        if len(person.image_paths) == 0:
            print(f"[Test] Skipping {person_id}: no images")
            continue
        
        print(f"\n{'='*60}")
        print(f"[Test] Person: {person_id} ({person.name or 'unnamed'})")
        print(f"{'='*60}")
        
        first_image_path = person.image_paths[0]
        appearance = test_appearance_on_image(first_image_path)
        results[person_id] = appearance
    
    print(f"\n{'='*60}")
    print(f"[Test] Summary: {len([r for r in results.values() if r])} successful, {len([r for r in results.values() if not r])} failed")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test appearance extraction service")
    parser.add_argument("--image", type=str, help="Path to image file to test")
    parser.add_argument("--person", type=str, help="Person ID to test on")
    parser.add_argument("--all", action="store_true", help="Test on all people")
    
    args = parser.parse_args()
    
    try:
        if args.image:
            # Test on specific image file
            test_appearance_on_image(args.image)
        elif args.all:
            # Test on all people
            test_appearance_on_all_people()
        else:
            # Test on specific person or first person
            test_appearance_on_person(args.person)
    except KeyboardInterrupt:
        print("\n[Test] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Test] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

