# server/config.py
"""
Server Configuration
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "static" / "images"
EXPORTS_DIR = BASE_DIR / "static" / "exports"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Model paths
ARCFACE_MODEL = MODELS_DIR / "arcfaceresnet100-8.onnx"
HAAR_CASCADE = MODELS_DIR / "haarcascade_frontalface_default.xml"

# Database files
VECTOR_DB_FILE = DATA_DIR / "vectors.json"
METADATA_FILE = DATA_DIR / "metadata.json"

# Face recognition settings
SIMILARITY_THRESHOLD = 0.5
MIN_FACE_SIZE = (30, 30)
# Resize images to this max dimension before face detection (faster processing)
# Face detection runs on this smaller image, then crops/resizes to 112x112 for model
# Note: Lower values = faster but may miss small faces. 1600 preserves detail while still being faster.
# Increase to 2000+ if faces are being missed, decrease to 1200 for maximum speed.
DETECTION_MAX_WIDTH = 1600  # Resize to max 1600px width (maintains aspect ratio)

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Image storage
THUMBNAIL_SIZE = (150, 150)
MAX_IMAGES_PER_PERSON = 20  # Keep last N images per person
