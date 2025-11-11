"""
Image Processing Service
Utilities for image handling
"""
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import config

def bytes_to_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded bytes to OpenCV image"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img

def save_image(image: np.ndarray, person_id: str) -> str:
    """Save image and return path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{person_id}_{timestamp}.jpg"
    filepath = config.IMAGES_DIR / filename

    cv2.imwrite(str(filepath), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(filepath)

