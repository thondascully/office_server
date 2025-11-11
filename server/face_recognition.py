# server/face_recognition.py
"""
Face Recognition using ArcFace
"""
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Optional
import config

class FaceRecognizer:
    """ArcFace-based face recognition"""

    def __init__(self):
        print("[Face Recognition] Loading ArcFace model...")
        
        # Check if model files exist
        if not config.ARCFACE_MODEL.exists():
            raise FileNotFoundError(
                f"ArcFace model not found at {config.ARCFACE_MODEL}. "
                "Please download models using download_models.sh or ensure models are in the Docker image."
            )
        if not config.HAAR_CASCADE.exists():
            raise FileNotFoundError(
                f"Haar Cascade not found at {config.HAAR_CASCADE}. "
                "Please download models using download_models.sh or ensure models are in the Docker image."
            )
        
        try:
            self.arcface_session = ort.InferenceSession(
                str(config.ARCFACE_MODEL),
                providers=['CPUExecutionProvider']
            )
            self.face_cascade = cv2.CascadeClassifier(str(config.HAAR_CASCADE))
            print("[Face Recognition] Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load face recognition models: {e}")

    def detect_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and extract largest face
        Optimized: Resizes image before detection for faster processing
        """
        # Resize image for faster face detection (maintains aspect ratio)
        original_height, original_width = image.shape[:2]
        if original_width > config.DETECTION_MAX_WIDTH:
            scale = config.DETECTION_MAX_WIDTH / original_width
            new_width = config.DETECTION_MAX_WIDTH
            new_height = int(original_height * scale)
            detection_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            detection_image = image
            scale = 1.0

        # Face detection on resized image (much faster)
        gray = cv2.cvtColor(detection_image, cv2.COLOR_BGR2GRAY)
        
        # Scale min face size for resized image
        min_size_scaled = (int(config.MIN_FACE_SIZE[0] * scale), int(config.MIN_FACE_SIZE[1] * scale))
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=min_size_scaled
        )

        if len(faces) == 0:
            return None

        # Use largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        # Crop with padding (from resized image)
        padding = int(0.2 * max(w, h))
        y1 = max(0, y - padding)
        y2 = min(detection_image.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(detection_image.shape[1], x + w + padding)

        face_crop = detection_image[y1:y2, x1:x2]
        # Resize to model input size (112x112)
        face_resized = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LINEAR)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        return face_rgb

    def preprocess_face(self, face_rgb: np.ndarray) -> np.ndarray:
        """Preprocess for ArcFace"""
        face_normalized = (face_rgb.astype(np.float32) - 127.5) / 127.5
        face_tensor = np.transpose(face_normalized, (2, 0, 1))
        face_batch = np.expand_dims(face_tensor, axis=0)
        return face_batch

    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Get face embedding from image"""
        face_rgb = self.detect_face(image)
        if face_rgb is None:
            return None

        face_input = self.preprocess_face(face_rgb)

        input_name = self.arcface_session.get_inputs()[0].name
        output = self.arcface_session.run(None, {input_name: face_input})[0]

        embedding = output[0]
        embedding_normalized = embedding / np.linalg.norm(embedding)

        return embedding_normalized

    def compute_mean_embedding(self, images: List[np.ndarray], min_embeddings: int = 2) -> Optional[np.ndarray]:
        """
        Compute mean embedding from multiple images
        Optimized: stops early once we have enough good embeddings (min_embeddings)
        """
        embeddings = []
        
        # Process images until we have enough good embeddings
        for img in images:
            embedding = self.get_embedding(img)
            if embedding is not None:
                embeddings.append(embedding)
                # Early stop: if we have enough embeddings, we can stop processing
                # This speeds up processing when we have 5+ images
                if len(embeddings) >= min_embeddings and len(embeddings) >= 3:
                    # For 5 images, stop after 3-4 good embeddings (faster, still accurate)
                    if len(images) <= 5:
                        break

        if len(embeddings) == 0:
            return None

        mean_embedding = np.mean(embeddings, axis=0)
        mean_embedding_normalized = mean_embedding / np.linalg.norm(mean_embedding)

        return mean_embedding_normalized


# Global instance
face_recognizer = FaceRecognizer()
