"""
Image Processing Service
Utilities for image handling
"""
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
import config

def bytes_to_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded bytes to OpenCV image"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img

def draw_bounding_box(image: np.ndarray, bbox: tuple, color: tuple = (0, 255, 0), thickness: int = 2, confidence: Optional[float] = None) -> np.ndarray:
    """Draw a bounding box on an image
    Args:
        image: OpenCV image (BGR format)
        bbox: (x, y, w, h) bounding box coordinates
        color: BGR color tuple (default: green)
        thickness: Line thickness
        confidence: Optional confidence score to display as text
    Returns:
        Image with bounding box drawn
    """
    x, y, w, h = bbox
    # Draw rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    
    # Add confidence text if provided
    if confidence is not None:
        text = f"{confidence:.2%}"
        # Calculate text size for positioning
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
        
        # Position text above the bounding box
        text_x = x
        text_y = max(y - 5, text_height + 5)  # Position above box, or at top if too close
        
        # Draw background rectangle for text
        cv2.rectangle(image, 
                     (text_x, text_y - text_height - 5), 
                     (text_x + text_width + 5, text_y + baseline), 
                     color, -1)
        
        # Draw text
        cv2.putText(image, text, (text_x + 2, text_y - 2), 
                   font, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)
    
    return image

def save_image(image: np.ndarray, person_id: str, bbox: Optional[tuple] = None, confidence: Optional[float] = None) -> str:
    """Save image and return path
    Args:
        image: OpenCV image (BGR format)
        person_id: Person identifier
        bbox: Optional (x, y, w, h) bounding box to draw on image
        confidence: Optional confidence score to display on bounding box
    """
    # Create a copy to avoid modifying the original
    img_to_save = image.copy()
    
    # Draw bounding box if provided
    if bbox is not None:
        img_to_save = draw_bounding_box(img_to_save, bbox, confidence=confidence)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{person_id}_{timestamp}.jpg"
    filepath = config.IMAGES_DIR / filename

    cv2.imwrite(str(filepath), img_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(filepath)

