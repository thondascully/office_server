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
        self.arcface_session = ort.InferenceSession(
            str(config.ARCFACE_MODEL),
            providers=['CPUExecutionProvider']
        )

        self.face_cascade = cv2.CascadeClassifier(str(config.HAAR_CASCADE))
        print("[Face Recognition] Model loaded successfully")

    def detect_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect and extract largest face"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=config.MIN_FACE_SIZE
        )

        if len(faces) == 0:
            return None

        # Use largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        # Crop with padding
        padding = int(0.2 * max(w, h))
        y1 = max(0, y - padding)
        y2 = min(image.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(image.shape[1], x + w + padding)

        face_crop = image[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, (112, 112))
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

    def compute_mean_embedding(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """Compute mean embedding from multiple images"""
        embeddings = []

        for img in images:
            embedding = self.get_embedding(img)
            if embedding is not None:
                embeddings.append(embedding)

        if len(embeddings) == 0:
            return None

        mean_embedding = np.mean(embeddings, axis=0)
        mean_embedding_normalized = mean_embedding / np.linalg.norm(mean_embedding)

        # Log embedding generation (only if verbose logging needed)
        # print(f"[Face Recognition] Generated mean embedding from {len(embeddings)}/{len(images)} images")
        return mean_embedding_normalized


# Global instance
face_recognizer = FaceRecognizer()
