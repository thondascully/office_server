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
        print("[Face Recognition] Loading models...")
        
        # Check if model files exist
        if not config.ARCFACE_MODEL.exists():
            raise FileNotFoundError(
                f"ArcFace model not found at {config.ARCFACE_MODEL}. "
                "Please download models using download_models.sh or ensure models are in the Docker image."
            )
        
        try:
            self.arcface_session = ort.InferenceSession(
                str(config.ARCFACE_MODEL),
                providers=['CPUExecutionProvider']
            )
            
            # RetinaFace is optional (deprecated - unreliable, tiling is preferred)
            self.use_retinaface = False
            if config.RETINAFACE_MODEL.exists():
                try:
                    print("[Face Recognition] Loading RetinaFace detector (optional, deprecated)...")
                    self.retinaface_session = ort.InferenceSession(
                        str(config.RETINAFACE_MODEL),
                        providers=['CPUExecutionProvider']
                    )
                    self.use_retinaface = True
                    self.retinaface_input_size = 640
                    print("[Face Recognition] RetinaFace loaded (will use tiling instead)")
                except Exception as e:
                    print(f"[Face Recognition] Failed to load RetinaFace: {e}")
                    print("[Face Recognition] Continuing without RetinaFace (tiling will be used)")
                    self.use_retinaface = False
            else:
                print("[Face Recognition] RetinaFace not found (optional, tiling will be used)")
            
            # Load UltraFace for tiling approach (optional)
            self.use_ultraface = False
            if config.ULTRAFACE_MODEL.exists():
                print("[Face Recognition] Loading UltraFace detector for tiling...")
                try:
                    self.ultraface_session = ort.InferenceSession(
                        str(config.ULTRAFACE_MODEL),
                        providers=['CPUExecutionProvider']
                    )
                    self.use_ultraface = True
                    self.ultraface_input_size = 640
                    print("[Face Recognition] UltraFace loaded successfully")
                except Exception as e:
                    print(f"[Face Recognition] Failed to load UltraFace model: {e}")
                    print("[Face Recognition] UltraFace model file may be corrupted. Please re-download it.")
                    print("[Face Recognition] Tiling will use Haar Cascade fallback")
                    self.use_ultraface = False
            else:
                print("[Face Recognition] UltraFace not found, tiling will use Haar Cascade")
            
            # Load Haar Cascade for tiling fallback (if UltraFace not available)
            if not self.use_ultraface:
                if config.HAAR_CASCADE.exists():
                    print("[Face Recognition] Loading Haar Cascade for tiling fallback...")
                    self.face_cascade = cv2.CascadeClassifier(str(config.HAAR_CASCADE))
                    if self.face_cascade.empty():
                        print("[Face Recognition] WARNING: Haar Cascade file is invalid")
                        self.face_cascade = None
                    else:
                        print("[Face Recognition] Haar Cascade loaded for tiling")
                else:
                    print("[Face Recognition] WARNING: Neither UltraFace nor Haar Cascade available for tiling")
                    self.face_cascade = None
            
            print("[Face Recognition] All models loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load face recognition models: {e}")

    def _detect_face_retinaface(self, image: np.ndarray) -> Optional[tuple]:
        """Detect face using RetinaFace ResNet50 model"""
        original_height, original_width = image.shape[:2]
        
        # RetinaFace expects input size 640x640
        # Resize image maintaining aspect ratio and pad to square
        scale = min(self.retinaface_input_size / original_width, self.retinaface_input_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square (640x640)
        padded_image = np.zeros((self.retinaface_input_size, self.retinaface_input_size, 3), dtype=np.uint8)
        padded_image[:new_height, :new_width] = resized_image
        
        # Preprocess for RetinaFace (BGR to RGB, normalize to [-1, 1])
        input_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32)
        input_image = (input_image - 127.5) / 128.0
        input_image = np.transpose(input_image, (2, 0, 1))  # HWC to CHW
        input_batch = np.expand_dims(input_image, axis=0)  # Add batch dimension
        
        # Run inference
        input_name = self.retinaface_session.get_inputs()[0].name
        outputs = self.retinaface_session.run(None, {input_name: input_batch})
        
        # RetinaFace outputs: [loc (boxes), conf (scores), landms (landmarks)]
        # Based on debug output:
        #   Output 0: shape=(1, 16800, 4) - boxes (loc) in format [cx, cy, w, h] or [x1, y1, x2, y2]
        #   Output 1: shape=(1, 16800, 2) - scores (conf) with 2 classes [background, face]
        #   Output 2: shape=(1, 16800, 10) - landmarks (5 points * 2 coords)
        
        # Extract boxes and scores (remove batch dimension)
        boxes_raw = outputs[0][0]  # Shape: [16800, 4]
        scores_raw = outputs[1][0]  # Shape: [16800, 2] - [background_score, face_score]
        
        # Get face confidence scores (second class, index 1)
        # NOTE: The model outputs are already probabilities (softmax already applied)
        # So we can directly use the second column as face confidence
        face_scores = scores_raw[:, 1]  # Face class probabilities (already softmaxed)
        
        # Generate anchors first (needed for decoding)
        feature_sizes = [80, 40, 20]
        anchor_scales = [16, 32]
        anchors = []
        for feat_size in feature_sizes:
            stride = self.retinaface_input_size // feat_size
            for scale in anchor_scales:
                for y in range(feat_size):
                    for x in range(feat_size):
                        cx = (x + 0.5) * stride
                        cy = (y + 0.5) * stride
                        anchors.append([cx, cy, scale, scale])
        
        # Debug: print score statistics
        max_score = np.max(face_scores)
        mean_score = np.mean(face_scores)
        print(f"[Face Recognition] RetinaFace: Score stats - max: {max_score:.4f}, mean: {mean_score:.4f}, anchors: {len(anchors)}")
        
        # Filter by confidence threshold and box size
        # First, decode top detections to find ones with reasonable size
        sorted_indices = np.argsort(face_scores)[::-1]
        top_n_check = min(500, len(face_scores))  # Check top 500 detections
        
        valid_boxes = []
        valid_scores = []
        valid_indices_list = []
        
        for idx in sorted_indices[:top_n_check]:
            box_candidate = boxes_raw[idx]
            score_candidate = face_scores[idx]
            
            # Decode box to check size
            if len(anchors) == len(boxes_raw):
                anchor = anchors[idx]
                anchor_cx, anchor_cy, anchor_w, anchor_h = anchor
                dx, dy, dw, dh = box_candidate[:4]
                cx = anchor_cx + dx * anchor_w
                cy = anchor_cy + dy * anchor_h
                w = anchor_w * np.exp(dw)
                h = anchor_h * np.exp(dh)
                
                # Convert to corner format and check bounds
                x1_temp = cx - w / 2
                y1_temp = cy - h / 2
                x2_temp = cx + w / 2
                y2_temp = cy + h / 2
                
                # Filter: reasonable size (20-500 pixels in 640x640 space), confidence, and within image bounds
                # The image is resized to (new_width, new_height) and placed at top-left of 640x640
                # So valid coordinates are: x in [0, new_width], y in [0, new_height]
                within_bounds = (x1_temp >= 0 and y1_temp >= 0 and 
                                x2_temp <= new_width and y2_temp <= new_height)
                
                # Lower confidence threshold and wider size range to debug
                # TODO: The model seems to have very low confidence scores - may need different preprocessing
                if w >= 20 and h >= 20 and w <= 500 and h <= 500 and score_candidate > 0.001 and within_bounds:
                    valid_boxes.append(box_candidate)
                    valid_scores.append(score_candidate)
                    valid_indices_list.append(idx)
        
        if len(valid_boxes) == 0:
            print(f"[Face Recognition] RetinaFace: No valid faces found (checked {top_n_check} top detections)")
            return None
        
        # Use the best valid detection
        best_idx = np.argmax(valid_scores)
        box = valid_boxes[best_idx]
        confidence = valid_scores[best_idx]
        original_best_idx = valid_indices_list[best_idx]
        
        print(f"[Face Recognition] RetinaFace: Found {len(valid_boxes)} valid face(s) out of {top_n_check} checked, using best (confidence: {confidence:.4f}, idx: {original_best_idx})")
        print(f"[Face Recognition] RetinaFace: Box raw values: {box[:4]}")
        
        # Decode the selected box using anchor
        if len(anchors) != len(boxes_raw):
            print(f"[Face Recognition] RetinaFace: Anchor count mismatch ({len(anchors)} vs {len(boxes_raw)}), using fallback")
            # Fallback: assume boxes might be in a different format
            box_coords = box[:4]
            dx, dy, dw, dh = box_coords
            
            # Try different interpretations
            if abs(dx) < 10 and abs(dy) < 10 and abs(dw) < 10 and abs(dh) < 10:
                # Small values - likely normalized or need anchor decoding but we don't have the right anchor
                # Try to use a reasonable anchor estimate
                # Find approximate anchor position from index
                # 16800 anchors, try to estimate position
                anchor_idx = original_best_idx
                # Rough estimate: anchors are distributed across the image
                approx_x = (anchor_idx % 160) * 4  # Rough estimate
                approx_y = (anchor_idx // 160) * 4
                approx_w = 32
                approx_h = 32
                
                cx = approx_x + dx * approx_w
                cy = approx_y + dy * approx_h
                w = approx_w * np.exp(dw) if dw < 10 else approx_w
                h = approx_h * np.exp(dh) if dh < 10 else approx_h
                
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
            else:
                # Large values - might be pixel coordinates or center format
                if abs(dx) < self.retinaface_input_size and abs(dy) < self.retinaface_input_size:
                    # Try as corner coordinates
                    x1, y1, x2, y2 = dx, dy, dw, dh
                else:
                    # Try as center format
                    cx, cy, w_box, h_box = dx, dy, dw, dh
                    x1 = cx - w_box / 2
                    y1 = cy - h_box / 2
                    x2 = cx + w_box / 2
                    y2 = cy + h_box / 2
        else:
            # Decode using proper anchor
            anchor = anchors[original_best_idx]
            anchor_cx, anchor_cy, anchor_w, anchor_h = anchor
            
            dx, dy, dw, dh = box[:4]
            
            # Standard RetinaFace decoding formula
            cx = anchor_cx + dx * anchor_w
            cy = anchor_cy + dy * anchor_h
            w = anchor_w * np.exp(dw)
            h = anchor_h * np.exp(dh)
            
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            print(f"[Face Recognition] RetinaFace: Decoded from anchor {original_best_idx}: anchor=({anchor_cx:.1f},{anchor_cy:.1f},{anchor_w:.1f},{anchor_h:.1f}), offsets=({dx:.3f},{dy:.3f},{dw:.3f},{dh:.3f})")
            print(f"[Face Recognition] RetinaFace: Decoded box: ({x1:.1f},{y1:.1f}) to ({x2:.1f},{y2:.1f}), size={w:.1f}x{h:.1f}")
        
        # Clamp to image bounds
        x1 = max(0, min(x1, self.retinaface_input_size))
        y1 = max(0, min(y1, self.retinaface_input_size))
        x2 = max(0, min(x2, self.retinaface_input_size))
        y2 = max(0, min(y2, self.retinaface_input_size))
        
        # Ensure x1 < x2 and y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Scale back to original image coordinates
        # Coordinates are in 640x640 space (padded image)
        # The image was resized by scale factor, then placed at top-left of 640x640
        # So coordinates in 640x640 space need to be scaled by 1/scale to get original coordinates
        
        # scale = min(640/original_width, 640/original_height)
        # So to go back: multiply by original_width/640 or original_height/640
        # But we used the minimum scale, so we need to use the corresponding dimension
        if scale == self.retinaface_input_size / original_width:
            # Width was the limiting factor
            scale_back_x = original_width / self.retinaface_input_size
            scale_back_y = original_height / new_height  # Use actual resized height
        else:
            # Height was the limiting factor  
            scale_back_x = original_width / new_width  # Use actual resized width
            scale_back_y = original_height / self.retinaface_input_size
        
        x1_orig = int(x1 * scale_back_x)
        y1_orig = int(y1 * scale_back_y)
        x2_orig = int(x2 * scale_back_x)
        y2_orig = int(y2 * scale_back_y)
        
        # Clamp to image boundaries
        x1_orig = max(0, min(x1_orig, original_width))
        y1_orig = max(0, min(y1_orig, original_height))
        x2_orig = max(0, min(x2_orig, original_width))
        y2_orig = max(0, min(y2_orig, original_height))
        
        w = x2_orig - x1_orig
        h = y2_orig - y1_orig
        
        print(f"[Face Recognition] RetinaFace: Scaled box from 640x640: ({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f}) to original: ({x1_orig},{y1_orig})-({x2_orig},{y2_orig}), size={w}x{h}")
        
        # Validate size
        if w < 30 or h < 30:
            print(f"[Face Recognition] RetinaFace: Face too small ({w}x{h})")
            return None
        
        # Crop face from original image with some padding
        padding = int(0.1 * max(w, h))  # 10% padding
        x1_crop = max(0, x1_orig - padding)
        y1_crop = max(0, y1_orig - padding)
        x2_crop = min(original_width, x2_orig + padding)
        y2_crop = min(original_height, y2_orig + padding)
        
        face_crop = image[y1_crop:y2_crop, x1_crop:x2_crop]
        
        if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            print(f"[Face Recognition] RetinaFace: Invalid face crop")
            return None
        
        # Resize to 112x112 for ArcFace
        face_resized = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LINEAR)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Return bounding box in original image coordinates (without padding for display)
        # Also return confidence for display purposes
        bbox = (x1_orig, y1_orig, w, h)
        print(f"[Face Recognition] RetinaFace: Selected face at ({x1_orig}, {y1_orig}), size {w}x{h}, confidence: {confidence:.4f}")
        
        # Store confidence in the return tuple (we'll need to modify the return format)
        # For now, we'll return (face_rgb, bbox, confidence) but need to update callers
        return (face_rgb, bbox, confidence)
    
    def _detect_face_haar(self, image: np.ndarray) -> Optional[tuple]:
        """Detect face using Haar Cascade (DEPRECATED - not used, kept for reference)
        Note: Haar Cascade is now used directly in tiling approach, not as standalone method
        """
        # Resize image for faster face detection (maintains aspect ratio)
        original_height, original_width = image.shape[:2]
        
        # Only resize if DETECTION_MAX_WIDTH is set and image is larger
        if config.DETECTION_MAX_WIDTH and original_width > config.DETECTION_MAX_WIDTH:
            scale = config.DETECTION_MAX_WIDTH / original_width
            new_width = config.DETECTION_MAX_WIDTH
            new_height = int(original_height * scale)
            detection_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            print(f"[Face Recognition] Resized: {original_width}x{original_height} -> {new_width}x{new_height} (scale: {scale:.3f})")
        else:
            detection_image = image
            scale = 1.0
            print(f"[Face Recognition] Using original size: {original_width}x{original_height}")

        # Face detection on resized image (much faster)
        gray = cv2.cvtColor(detection_image, cv2.COLOR_BGR2GRAY)
        
        # Scale min face size for resized image
        # After resizing, faces become smaller proportionally, so min size should scale down
        # But don't go below 15px (too small = false positives)
        if scale < 1.0:
            min_size_scaled = (
                max(15, int(config.MIN_FACE_SIZE[0] * scale)), 
                max(15, int(config.MIN_FACE_SIZE[1] * scale))
            )
        else:
            min_size_scaled = config.MIN_FACE_SIZE
        
        print(f"[Face Recognition] Detection: image={detection_image.shape[1]}x{detection_image.shape[0]}, min_size={min_size_scaled}")
        
        # Try detection with multiple strategies (balanced parameters first)
        # Strategy 1: Balanced parameters (good balance of speed and accuracy)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,  # Increased from 3 - requires more neighbors = fewer false positives
            minSize=min_size_scaled,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        print(f"[Face Recognition] First attempt (balanced): found {len(faces)} faces")

        # Strategy 2: If no faces, try with even smaller min size
        if len(faces) == 0:
            min_size_fallback = (max(10, int(min_size_scaled[0] * 0.5)), max(10, int(min_size_scaled[1] * 0.5)))
            print(f"[Face Recognition] Retrying with min_size={min_size_fallback}")
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,  # Smaller steps = more thorough
                minNeighbors=2,  # Very lenient
                minSize=min_size_fallback,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            print(f"[Face Recognition] Fallback attempt: found {len(faces)} faces")
        
        # Strategy 3: Last resort - but still require reasonable face size
        if len(faces) == 0:
            min_size_last_resort = (max(20, int(min_size_scaled[0] * 0.7)), max(20, int(min_size_scaled[1] * 0.7)))
            print(f"[Face Recognition] Last resort: trying with min_size={min_size_last_resort}")
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=2,  # Still require at least 2 neighbors (not 1 - too lenient)
                minSize=min_size_last_resort,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            print(f"[Face Recognition] Last resort attempt: found {len(faces)} faces")

        if len(faces) == 0:
            print(f"[Face Recognition] No faces detected in {detection_image.shape[1]}x{detection_image.shape[0]} image")
            return None

        # Filter faces by quality - faces should have reasonable aspect ratio (not too wide/tall)
        # Typical face aspect ratio is roughly 0.7-1.3 (width/height)
        valid_faces = []
        for (fx, fy, fw, fh) in faces:
            aspect_ratio = fw / fh if fh > 0 else 0
            # Faces are typically roughly square to slightly tall (not extremely wide or tall)
            if 0.6 <= aspect_ratio <= 1.5:
                valid_faces.append((fx, fy, fw, fh))
            else:
                print(f"[Face Recognition] Rejected face candidate: aspect ratio {aspect_ratio:.2f} (expected 0.6-1.5)")
        
        if len(valid_faces) == 0:
            print(f"[Face Recognition] No valid faces after aspect ratio filtering (found {len(faces)} candidates)")
            return None
        
        # Use largest valid face
        largest_face = max(valid_faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Additional validation: face should be reasonably sized
        face_area = w * h
        image_area = detection_image.shape[0] * detection_image.shape[1]
        face_ratio = face_area / image_area if image_area > 0 else 0
        
        # Face should be at least 0.5% of image (increased from 0.1% to reject very small false positives)
        # and at most 50% (reasonable bounds)
        if face_ratio < 0.005 or face_ratio > 0.5:
            print(f"[Face Recognition] Rejected face: size ratio {face_ratio:.4f} (too small <0.5% or too large >50%)")
            return None
        
        # Additional check: absolute face size should be reasonable
        # Very small faces (like 91x91) are often false positives
        # After resizing, we want faces to be at least 60x60 pixels in the detection image
        if w < 60 or h < 60:
            print(f"[Face Recognition] Rejected face: too small ({w}x{h} pixels, minimum 60x60)")
            return None
        
        print(f"[Face Recognition] Selected face: {w}x{h} at ({x},{y}), aspect={w/h:.2f}, area_ratio={face_ratio:.4f}")

        # Crop with padding (from resized image)
        padding = int(0.2 * max(w, h))
        y1 = max(0, y - padding)
        y2 = min(detection_image.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(detection_image.shape[1], x + w + padding)

        face_crop = detection_image[y1:y2, x1:x2]
        
        # Final validation: ensure crop is valid and not empty
        if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            print(f"[Face Recognition] Invalid face crop: {face_crop.shape}")
            return None
        
        # Additional check: face crop should have reasonable color distribution (not all one color)
        # This helps catch cases where we're detecting non-face regions like hands/arms
        face_gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        gray_std = np.std(face_gray_crop)
        if gray_std < 15:  # Very low variance = mostly uniform color = probably not a face
            print(f"[Face Recognition] Rejected: face crop has low variance ({gray_std:.2f}), likely not a real face (hands/arms/background)")
            return None
        
        # Resize to model input size (112x112)
        face_resized = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LINEAR)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        # Scale bounding box back to original image coordinates
        # The bbox (x, y, w, h) is in detection_image coordinates, need to scale back
        bbox_original = (
            int(x / scale),
            int(y / scale),
            int(w / scale),
            int(h / scale)
        )

        return (face_rgb, bbox_original)
    
    def _detect_faces_ultraface(self, image: np.ndarray, confidence_threshold: float = 0.6) -> List[tuple]:
        """Detect faces using UltraFace model
        Returns: List of (bbox, confidence) tuples where bbox is (x, y, w, h) in image coordinates
        """
        if not self.use_ultraface:
            return []
        
        original_height, original_width = image.shape[:2]
        
        # UltraFace expects 640x640 input
        scale = min(self.ultraface_input_size / original_width, self.ultraface_input_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize and pad to square
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((self.ultraface_input_size, self.ultraface_input_size, 3), dtype=np.uint8)
        padded[:new_height, :new_width] = resized
        
        # Preprocess: BGR to RGB, normalize to [0, 1]
        input_img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(np.transpose(input_img, (2, 0, 1)), axis=0)
        
        # Run inference
        input_name = self.ultraface_session.get_inputs()[0].name
        outputs = self.ultraface_session.run(None, {input_name: input_tensor})
        
        # UltraFace outputs: [boxes, scores]
        # boxes: [1, num_detections, 4] in format [x1, y1, x2, y2] (normalized 0-1)
        # scores: [1, num_detections] confidence scores
        boxes = outputs[0][0]  # [num_detections, 4]
        scores = outputs[1][0]  # [num_detections]
        
        detections = []
        for i in range(len(scores)):
            if scores[i] > confidence_threshold:
                # Boxes are normalized [0, 1], convert to pixel coordinates in padded image
                x1_norm, y1_norm, x2_norm, y2_norm = boxes[i]
                x1 = int(x1_norm * self.ultraface_input_size)
                y1 = int(y1_norm * self.ultraface_input_size)
                x2 = int(x2_norm * self.ultraface_input_size)
                y2 = int(y2_norm * self.ultraface_input_size)
                
                # Clamp to resized image bounds (not padding)
                x1 = max(0, min(x1, new_width))
                y1 = max(0, min(y1, new_height))
                x2 = max(0, min(x2, new_width))
                y2 = max(0, min(y2, new_height))
                
                if x2 > x1 and y2 > y1:
                    # Scale back to original image coordinates
                    x1_orig = int(x1 / scale)
                    y1_orig = int(y1 / scale)
                    w_orig = int((x2 - x1) / scale)
                    h_orig = int((y2 - y1) / scale)
                    
                    if w_orig >= 30 and h_orig >= 30:  # Minimum face size
                        detections.append(((x1_orig, y1_orig, w_orig, h_orig), float(scores[i])))
        
        return detections
    
    def _nms(self, detections: List[tuple], iou_threshold: float = 0.4) -> List[tuple]:
        """Non-Maximum Suppression to merge overlapping detections
        Args:
            detections: List of (bbox, confidence) where bbox is (x, y, w, h)
            iou_threshold: IoU threshold for merging (lower = more aggressive merging)
        Returns:
            Filtered list of (bbox, confidence) with overlapping detections removed
        """
        if len(detections) == 0:
            return []
        
        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x[1], reverse=True)
        
        def iou(box1, box2):
            """Calculate Intersection over Union for two boxes"""
            x1_1, y1_1, w1, h1 = box1
            x2_1, y2_1 = x1_1 + w1, y1_1 + h1
            
            x1_2, y1_2, w2, h2 = box2
            x2_2, y2_2 = x1_2 + w2, y1_2 + h2
            
            # Calculate intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        kept = []
        while len(detections) > 0:
            # Take the highest confidence detection
            current = detections.pop(0)
            kept.append(current)
            
            # Remove all detections that overlap significantly with current
            current_box = current[0]
            detections = [
                det for det in detections
                if iou(current_box, det[0]) < iou_threshold
            ]
        
        return kept
    
    def _detect_face_tiling(self, image: np.ndarray, tripwire_zone: Optional[tuple] = None) -> Optional[tuple]:
        """Detect face using tiling approach with UltraFace or Haar Cascade
        Divides image into overlapping tiles to handle large images and prevent face splitting
        
        Args:
            image: Input image
            tripwire_zone: Optional tuple (outer_x, inner_x) to limit tiling to tripwire zone only
                          If None, processes entire image
        """
        original_height, original_width = image.shape[:2]
        
        # Tile configuration
        # Use 640x640 tiles (UltraFace input size) with 50% overlap to prevent face splitting
        tile_size = 640
        overlap = 0.5  # 50% overlap
        step = int(tile_size * (1 - overlap))  # Step size: 320 pixels
        
        # Determine tiling bounds based on tripwire zone
        if tripwire_zone:
            outer_x, inner_x = tripwire_zone
            # Only process tiles within tripwire zone (x between outer_x and inner_x)
            # Clamp to image bounds
            min_x = max(0, outer_x - tile_size)  # Start a bit before to catch overlaps
            max_x = min(original_width, inner_x + tile_size)  # End a bit after to catch overlaps
            x_range = range(min_x, max_x, step)
            print(f"[Face Recognition] Tiling: image={original_width}x{original_height}, tripwire zone: x=[{outer_x}, {inner_x}], tiling x=[{min_x}, {max_x}]")
        else:
            x_range = range(0, original_width, step)
            print(f"[Face Recognition] Tiling: image={original_width}x{original_height}, tile_size={tile_size}, step={step}, overlap={overlap*100:.0f}%")
        
        all_detections = []
        
        # Process tiles (only within tripwire zone if specified)
        for y in range(0, original_height, step):
            for x in x_range:
                # Calculate tile bounds (with overlap)
                x_end = min(x + tile_size, original_width)
                y_end = min(y + tile_size, original_height)
                
                # Adjust start if we're at the edge (to ensure full tile_size)
                if x_end - x < tile_size:
                    x = max(0, x_end - tile_size)
                if y_end - y < tile_size:
                    y = max(0, y_end - tile_size)
                
                tile = image[y:y_end, x:x_end]
                
                if tile.size == 0:
                    continue
                
                # Detect faces in tile
                tile_detections = []
                if self.use_ultraface:
                    tile_detections = self._detect_faces_ultraface(tile, confidence_threshold=0.5)
                elif hasattr(self, 'face_cascade') and self.face_cascade is not None:
                    # Fallback to Haar Cascade
                    gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(
                        gray_tile,
                        scaleFactor=1.1,
                        minNeighbors=4,
                        minSize=(30, 30)
                    )
                    for (fx, fy, fw, fh) in faces:
                        tile_detections.append(((fx, fy, fw, fh), 0.8))  # Default confidence for Haar
                else:
                    # No detector available for this tile, skip
                    continue
                
                # Adjust coordinates to original image space
                for (bbox, conf) in tile_detections:
                    tx, ty, tw, th = bbox
                    # Translate to original image coordinates
                    orig_x = x + tx
                    orig_y = y + ty
                    all_detections.append(((orig_x, orig_y, tw, th), conf))
        
        if len(all_detections) == 0:
            print("[Face Recognition] Tiling: No faces detected in any tile")
            return None
        
        print(f"[Face Recognition] Tiling: Found {len(all_detections)} detections across all tiles")
        
        # Apply NMS to merge overlapping detections
        filtered_detections = self._nms(all_detections, iou_threshold=0.4)
        print(f"[Face Recognition] Tiling: After NMS: {len(filtered_detections)} detections")
        
        if len(filtered_detections) == 0:
            return None
        
        # Select the highest confidence detection
        best_detection = max(filtered_detections, key=lambda x: x[1])
        bbox, confidence = best_detection
        x, y, w, h = bbox
        
        # Crop face from original image with padding
        padding = int(0.1 * max(w, h))
        x1_crop = max(0, x - padding)
        y1_crop = max(0, y - padding)
        x2_crop = min(original_width, x + w + padding)
        y2_crop = min(original_height, y + h + padding)
        
        face_crop = image[y1_crop:y2_crop, x1_crop:x2_crop]
        
        if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            return None
        
        # Resize to 112x112 for ArcFace
        face_resized = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LINEAR)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        print(f"[Face Recognition] Tiling: Selected face at ({x}, {y}), size {w}x{h}, confidence: {confidence:.4f}")
        
        return (face_rgb, bbox, confidence)
    
    def detect_face(self, image: np.ndarray, tripwire_zone: Optional[tuple] = None) -> Optional[tuple]:
        """
        Detect and extract largest face
        Uses tiling approach (primary method) for all images
        Returns: (face_crop, bbox, confidence) where bbox is (x, y, w, h) in original image coordinates
        """
        # Use tiling as primary method for all images
        # Tiling works well for both large and small images
        print("[Face Recognition] Using tiling approach")
        try:
            result = self._detect_face_tiling(image, tripwire_zone=tripwire_zone)
            if result is None:
                print("[Face Recognition] Tiling failed to detect face")
                # Optional: Try RetinaFace as last resort (if available, but it's unreliable)
                if self.use_retinaface:
                    print("[Face Recognition] Trying RetinaFace as last resort (may be unreliable)")
                    try:
                        result = self._detect_face_retinaface(image)
                        if result is not None:
                            print("[Face Recognition] RetinaFace found face (low confidence expected)")
                    except Exception as e:
                        print(f"[Face Recognition] RetinaFace error: {e}")
            return result
        except Exception as e:
            print(f"[Face Recognition] Tiling error: {e}")
            import traceback
            traceback.print_exc()
            # Last resort: try RetinaFace if available
            if self.use_retinaface:
                try:
                    print("[Face Recognition] Trying RetinaFace as last resort after error")
                    return self._detect_face_retinaface(image)
                except:
                    pass
            return None

    def preprocess_face(self, face_rgb: np.ndarray) -> np.ndarray:
        """Preprocess for ArcFace"""
        face_normalized = (face_rgb.astype(np.float32) - 127.5) / 127.5
        face_tensor = np.transpose(face_normalized, (2, 0, 1))
        face_batch = np.expand_dims(face_tensor, axis=0)
        return face_batch

    def get_embedding(self, image: np.ndarray, tripwire_zone: Optional[tuple] = None) -> Optional[tuple]:
        """Get face embedding from image
        Returns: (embedding, bbox, confidence) where bbox is (x, y, w, h) in original image coordinates, 
                 confidence is the detection confidence score, or None if no face detected
        """
        result = self.detect_face(image, tripwire_zone=tripwire_zone)
        if result is None:
            return None
        
        face_rgb, bbox, confidence = result

        face_input = self.preprocess_face(face_rgb)

        input_name = self.arcface_session.get_inputs()[0].name
        output = self.arcface_session.run(None, {input_name: face_input})[0]

        embedding = output[0]
        
        # Validate embedding is not corrupted (should have reasonable magnitude)
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm < 1e-6:
            print(f"[Face Recognition] WARNING: Embedding norm too small ({embedding_norm:.6f}), possible corruption")
            return None
        
        embedding_normalized = embedding / embedding_norm
        
        # Validate normalized embedding (should be unit vector)
        normalized_norm = np.linalg.norm(embedding_normalized)
        if abs(normalized_norm - 1.0) > 0.01:
            print(f"[Face Recognition] WARNING: Normalized embedding norm is {normalized_norm:.6f} (expected ~1.0)")
        
        return (embedding_normalized, bbox, confidence)

    def compute_mean_embedding(self, images: List[np.ndarray], min_embeddings: int = 2, tripwire_zone: Optional[tuple] = None) -> Optional[tuple]:
        """
        Compute mean embedding from multiple images
        Optimized: stops early once we have enough good embeddings (min_embeddings)
        Returns: (mean_embedding, bbox_map) where bbox_map is a dict mapping image_index -> (bbox, confidence),
                 where bbox is (x, y, w, h) and confidence is float, or None if no faces detected
        """
        embeddings = []
        bbox_map = {}  # Maps image index to (bbox, confidence) tuple
        failed_count = 0
        
        # Process images until we have enough good embeddings
        for i, img in enumerate(images):
            result = self.get_embedding(img, tripwire_zone=tripwire_zone)
            if result is not None:
                embedding, bbox, confidence = result
                embeddings.append(embedding)
                bbox_map[i] = (bbox, confidence)
                # Early stop: if we have enough embeddings, we can stop processing
                # This speeds up processing when we have 5+ images
                if len(embeddings) >= min_embeddings and len(embeddings) >= 3:
                    # For 5 images, stop after 3-4 good embeddings (faster, still accurate)
                    if len(images) <= 5:
                        break
            else:
                failed_count += 1

        if len(embeddings) == 0:
            # Log why no embeddings were found
            print(f"[Face Recognition] No faces detected in any of {len(images)} images (failed: {failed_count})")
            return None

        mean_embedding = np.mean(embeddings, axis=0)
        mean_embedding_normalized = mean_embedding / np.linalg.norm(mean_embedding)

        return (mean_embedding_normalized, bbox_map)


# Global instance
face_recognizer = FaceRecognizer()
