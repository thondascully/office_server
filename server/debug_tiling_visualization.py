"""
Debug script to visualize tiling approach for face detection
Shows tiles, detections in each tile, and final merged result
Now includes tripwire zone visualization
"""
import cv2
import numpy as np
import sys
from pathlib import Path
from database import metadata_db
from face_recognition import face_recognizer
from services.rpi_manager import rpi_manager
import config

def visualize_tiling(image: np.ndarray, output_dir: Path, image_name: str, tripwire_zone: tuple = None):
    """Visualize tiling process for a single image with tripwire zone
    
    Args:
        image: Input image
        output_dir: Directory to save visualizations
        image_name: Name of the image file
        tripwire_zone: Optional tuple (outer_x, inner_x) to show tripwire boundaries
    """
    original_height, original_width = image.shape[:2]
    
    # Tile configuration (same as in face_recognition.py)
    tile_size = 640
    overlap = 0.5
    step = int(tile_size * (1 - overlap))  # 320 pixels
    
    # Create visualization images
    tile_vis = image.copy()  # Show all tiles
    detection_vis = image.copy()  # Show detections in tiles
    final_vis = image.copy()  # Show final result
    
    # Draw tripwire boundaries if provided
    if tripwire_zone:
        outer_x, inner_x = tripwire_zone
        # Draw tripwire lines (vertical lines)
        cv2.line(tile_vis, (outer_x, 0), (outer_x, original_height), (255, 165, 0), 3)  # Orange
        cv2.line(tile_vis, (inner_x, 0), (inner_x, original_height), (255, 165, 0), 3)  # Orange
        cv2.putText(tile_vis, "OUTER", (outer_x + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        cv2.putText(tile_vis, "INNER", (inner_x + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        
        # Draw on other visualizations too
        cv2.line(detection_vis, (outer_x, 0), (outer_x, original_height), (255, 165, 0), 3)
        cv2.line(detection_vis, (inner_x, 0), (inner_x, original_height), (255, 165, 0), 3)
        cv2.line(final_vis, (outer_x, 0), (outer_x, original_height), (255, 165, 0), 3)
        cv2.line(final_vis, (inner_x, 0), (inner_x, original_height), (255, 165, 0), 3)
        
        # Determine tiling bounds based on tripwire zone
        min_x = max(0, outer_x - tile_size)  # Start a bit before to catch overlaps
        max_x = min(original_width, inner_x + tile_size)  # End a bit after to catch overlaps
        x_range = range(min_x, max_x, step)
        print(f"  [Tiling] Tripwire zone: x=[{outer_x}, {inner_x}], processing x=[{min_x}, {max_x}]")
    else:
        x_range = range(0, original_width, step)
        print(f"  [Tiling] No tripwire zone, processing entire image")
    
    # Draw tile grid
    tile_colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    all_detections = []
    tile_num = 0
    
    # Process tiles (only within tripwire zone if specified)
    for y in range(0, original_height, step):
        for x in x_range:
            x_end = min(x + tile_size, original_width)
            y_end = min(y + tile_size, original_height)
            
            if x_end - x < tile_size:
                x = max(0, x_end - tile_size)
            if y_end - y < tile_size:
                y = max(0, y_end - tile_size)
            
            # Check if tile overlaps with tripwire zone (if specified)
            # For visualization, only show tiles whose center is within the tripwire zone
            if tripwire_zone:
                outer_x, inner_x = tripwire_zone
                # Calculate tile center
                tile_center_x = (x + x_end) / 2
                # Only show tiles whose center is within the tripwire zone
                tile_center_in_zone = (outer_x <= tile_center_x <= inner_x)
                if not tile_center_in_zone:
                    # Skip this tile - its center is outside the tripwire zone
                    continue
            
            tile = image[y:y_end, x:x_end]
            
            if tile.size == 0:
                continue
            
            # Draw tile boundary (only tiles that overlap tripwire zone are drawn)
            color = tile_colors[tile_num % len(tile_colors)]
            cv2.rectangle(tile_vis, (x, y), (x_end, y_end), color, 2)
            cv2.putText(tile_vis, f"T{tile_num}", (x + 5, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Detect faces in tile
            tile_detections = []
            if face_recognizer.use_ultraface:
                tile_detections = face_recognizer._detect_faces_ultraface(tile, confidence_threshold=0.5)
            elif hasattr(face_recognizer, 'face_cascade') and face_recognizer.face_cascade is not None:
                gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                faces = face_recognizer.face_cascade.detectMultiScale(
                    gray_tile,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(30, 30)
                )
                for (fx, fy, fw, fh) in faces:
                    tile_detections.append(((fx, fy, fw, fh), 0.8))
            
            # Draw detections in this tile
            for (bbox, conf) in tile_detections:
                tx, ty, tw, th = bbox
                orig_x = x + tx
                orig_y = y + ty
                
                # Draw on detection visualization
                cv2.rectangle(detection_vis, (orig_x, orig_y), 
                             (orig_x + tw, orig_y + th), color, 2)
                cv2.putText(detection_vis, f"{conf:.2f}", (orig_x, orig_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                all_detections.append(((orig_x, orig_y, tw, th), conf))
            
            tile_num += 1
    
    # Apply NMS to get final detections
    if len(all_detections) > 0:
        filtered_detections = face_recognizer._nms(all_detections, iou_threshold=0.4)
        
        # Draw final detection(s) - use highest confidence
        if len(filtered_detections) > 0:
            best_detection = max(filtered_detections, key=lambda x: x[1])
            bbox, confidence = best_detection
            x, y, w, h = bbox
            
            # Draw final bounding box in green with confidence
            cv2.rectangle(final_vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(final_vis, f"Final: {confidence:.2%}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Save visualizations
    base_name = Path(image_name).stem
    
    # 1. Tiles visualization
    tiles_path = output_dir / f"{base_name}_01_tiles.jpg"
    cv2.imwrite(str(tiles_path), tile_vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # 2. Detections in tiles
    detections_path = output_dir / f"{base_name}_02_detections_in_tiles.jpg"
    cv2.imwrite(str(detections_path), detection_vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # 3. Final result
    final_path = output_dir / f"{base_name}_03_final_bbox.jpg"
    cv2.imwrite(str(final_path), final_vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return len(filtered_detections) > 0 if len(all_detections) > 0 else False

def debug_tiling_visualization(rpi_id: str = "default"):
    """Create debug visualizations for all images in database with tripwire zone
    
    Args:
        rpi_id: RPi ID to load tripwire config from (default: "default")
    """
    print("[Debug Tiling] Starting tiling visualization for all images...")
    
    # Load tripwire configuration
    rpi_config = rpi_manager.load_config(rpi_id)
    tripwire_zone = None
    if rpi_config and 'tripwires' in rpi_config:
        tripwires = rpi_config['tripwires']
        outer_x = tripwires.get('outer_x', 800)
        inner_x = tripwires.get('inner_x', 1800)
        tripwire_zone = (outer_x, inner_x)
        print(f"[Debug Tiling] Using tripwire zone from {rpi_id}: x=[{outer_x}, {inner_x}]")
    else:
        print("[Debug Tiling] No tripwire config found, using default: x=[800, 1800]")
        tripwire_zone = (800, 1800)  # Default tripwires
    
    # Create output directory
    output_dir = config.IMAGES_DIR / "tiling_debug"
    output_dir.mkdir(exist_ok=True)
    print(f"[Debug Tiling] Output directory: {output_dir}")
    
    total_images = 0
    processed_images = 0
    failed_images = 0
    
    # Iterate through all people
    for person_id, person in metadata_db.people.items():
        print(f"\n[Debug Tiling] Processing person: {person_id} ({person.name or 'unnamed'})")
        print(f"  Images: {len(person.image_paths)}")
        
        for img_path in person.image_paths:
            total_images += 1
            img_file = Path(img_path)
            
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
                
                print(f"  [Processing] {img_file.name} ({image.shape[1]}x{image.shape[0]})")
                
                # Create visualizations with tripwire zone
                success = visualize_tiling(image, output_dir, img_file.name, tripwire_zone=tripwire_zone)
                
                if success:
                    processed_images += 1
                    print(f"  [OK] Created visualizations for: {img_file.name}")
                else:
                    print(f"  [WARNING] No faces detected in: {img_file.name}")
                    failed_images += 1
                
            except Exception as e:
                print(f"  [ERROR] Failed to process {img_path}: {e}")
                import traceback
                traceback.print_exc()
                failed_images += 1
                continue
    
    print(f"\n[Debug Tiling] Completed!")
    print(f"  Total images: {total_images}")
    print(f"  Successfully processed: {processed_images}")
    print(f"  Failed: {failed_images}")
    print(f"  Output directory: {output_dir}")

if __name__ == "__main__":
    try:
        debug_tiling_visualization()
    except KeyboardInterrupt:
        print("\n[Debug Tiling] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Debug Tiling] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

