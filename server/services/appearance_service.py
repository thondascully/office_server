"""
Appearance Extraction Service
Uses LLM to extract appearance features from images for back-detection matching
"""
import json
import base64
import cv2
import numpy as np
from typing import Optional, Dict
from pathlib import Path
from datetime import datetime
import requests

import config

class AppearanceService:
    """Extracts and matches appearance features using LLM"""
    
    def __init__(self, ollama_url: Optional[str] = None):
        self.ollama_url = ollama_url or config.OLLAMA_URL
        self.model = "phi3"  # Ollama model name - lightweight and available
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string"""
        # Encode image as JPEG
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        # Convert to base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    def extract_appearance(self, image: np.ndarray) -> Optional[Dict]:
        """Extract appearance features from image using LLM
        
        Args:
            image: OpenCV image (BGR format) containing a person
            
        Returns:
            Dictionary with appearance features:
            {
                "height_estimate": "tall/medium/short",
                "shirt_color": "color name",
                "hair_color": "color name",
                "hair_length": "short/medium/long",
                "gender": "male/female/unknown",
                "description": "concise description"
            }
            Returns None if extraction fails
        """
        try:
            # Convert image to base64
            image_base64 = self._image_to_base64(image)
            
            # Create prompt for LLM - focused on distinctive, matchable features
            prompt = """Analyze this image of a person and extract their appearance features for identification purposes.
Focus on DISTINCTIVE DETAILS that would help identify them even from behind or in different lighting.

Return ONLY a valid JSON object with the following structure (no markdown, no explanation, just JSON):
{
    "height_estimate": "tall" or "medium" or "short",
    "shirt_description": "detailed description including: primary color, any patterns/logos/text/designs, style (e.g., 'blue t-shirt with white text logo on chest', 'black polo with red stripes', 'solid red button-up'). Be specific about any text, logos, or distinctive patterns.",
    "shirt_color": "primary color name (e.g., blue, red, black, white) - note: lighting may affect this, so also describe patterns",
    "pants_description": "detailed description including: color, style (jeans, khakis, joggers, etc.), any logos/patterns, or 'not in picture' if pants are not visible",
    "pants_color": "color name or 'not in picture'",
    "hair_color": "color name (e.g., brown, black, blonde, dirty blonde, gray) - note: lighting/shadows may affect this",
    "hair_length": "short" or "medium" or "long",
    "gender": "male" or "female" or "unknown",
    "description": "concise 2-3 sentence description focusing on most distinctive features (clothing details, build, hair, etc.) that would help identify this person from behind"
}

IMPORTANT:
- For shirt_description: Include ANY text, logos, patterns, or distinctive features even if color is uncertain
- For pants: Use "not in picture" if pants are not visible
- Focus on features that are visible even when person is facing away (clothing details, build, hair from back)
- If lighting makes colors uncertain, describe patterns/details instead
- Be specific about clothing details (logos, text, patterns) as these are more reliable than colors alone"""
            
            # Call Ollama API with phi3
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_base64],
                    "stream": False,
                    "format": "json"
                },
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"[Appearance] Ollama API error: {response.status_code}")
                return None
            
            result = response.json()
            response_text = result.get("response", "")
            
            # Parse JSON response
            # Sometimes LLM wraps JSON in markdown or adds text, try to extract JSON
            try:
                # Try direct JSON parse
                appearance = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    json_text = response_text[json_start:json_end].strip()
                    appearance = json.loads(json_text)
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    json_text = response_text[json_start:json_end].strip()
                    appearance = json.loads(json_text)
                else:
                    # Try to find JSON object in text
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_text = response_text[json_start:json_end]
                        appearance = json.loads(json_text)
                    else:
                        raise ValueError("No JSON found in response")
            
            # Validate required fields
            required_fields = ["height_estimate", "shirt_description", "shirt_color", "pants_description", "pants_color", "hair_color", "hair_length", "gender", "description"]
            if not all(field in appearance for field in required_fields):
                print(f"[Appearance] Missing required fields in response: {appearance}")
                # Try to fill in missing fields with defaults
                if "pants_description" not in appearance:
                    appearance["pants_description"] = "not in picture"
                if "pants_color" not in appearance:
                    appearance["pants_color"] = "not in picture"
                if "shirt_description" not in appearance:
                    # Fallback: use shirt_color as description
                    appearance["shirt_description"] = appearance.get("shirt_color", "unknown")
                # Check again
                if not all(field in appearance for field in required_fields):
                    print(f"[Appearance] Still missing required fields: {appearance}")
                    return None
            
            print(f"[Appearance] Extracted appearance: {appearance}")
            
            # Log to file
            self._log_appearance_extraction(appearance, image)
            
            return appearance
            
        except Exception as e:
            print(f"[Appearance] Error extracting appearance: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def match_by_appearance(self, leave_image: np.ndarray, candidate_appearances: Dict[str, Dict]) -> Optional[tuple]:
        """Match leaving person by appearance features
        
        Optimized: First filters by shirt color (most reliable for back detection),
        then scores remaining candidates with full appearance matching.
        
        Args:
            leave_image: Image of person leaving (back to camera)
            candidate_appearances: Dict mapping person_id to appearance features
            
        Returns:
            Tuple (person_id, confidence_score) of best match, or None if no good match
        """
        try:
            # Extract appearance from leave image
            leave_appearance = self.extract_appearance(leave_image)
            if not leave_appearance:
                return None
            
            leave_shirt_description = leave_appearance.get("shirt_description", "").lower()
            leave_shirt_color = leave_appearance.get("shirt_color", "").lower()
            
            # Step 1: Filter by shirt description/color first (most reliable for back detection)
            # Prioritize distinctive details (logos, text, patterns) over color
            filtered_candidates = {}
            if leave_shirt_description or leave_shirt_color:
                print(f"[Appearance] Filtering candidates by shirt: description='{leave_shirt_description[:50]}...', color='{leave_shirt_color}'")
                for person_id, candidate_app in candidate_appearances.items():
                    candidate_shirt_desc = candidate_app.get("shirt_description", "").lower()
                    candidate_shirt_color = candidate_app.get("shirt_color", "").lower()
                    
                    # Match by distinctive details first (logos, text, patterns)
                    # Extract key words from descriptions (logos, text, patterns)
                    leave_keywords = set()
                    candidate_keywords = set()
                    
                    # Look for distinctive features (logos, text, patterns)
                    for word in ["logo", "text", "pattern", "stripes", "design", "print"]:
                        if word in leave_shirt_description:
                            leave_keywords.add(word)
                        if word in candidate_shirt_desc:
                            candidate_keywords.add(word)
                    
                    # Check for text/logo matches (e.g., "white text" in both)
                    text_match = False
                    if "text" in leave_shirt_description and "text" in candidate_shirt_desc:
                        # Try to extract text location (e.g., "on chest", "on front")
                        text_match = True  # If both have text, likely match
                    
                    # Color match (with tolerance for lighting differences)
                    color_match = False
                    if leave_shirt_color and candidate_shirt_color:
                        if leave_shirt_color == candidate_shirt_color or \
                           leave_shirt_color in candidate_shirt_color or \
                           candidate_shirt_color in leave_shirt_color:
                            color_match = True
                    
                    # Match if: distinctive features match OR color matches
                    if text_match or (len(leave_keywords) > 0 and leave_keywords.intersection(candidate_keywords)) or color_match:
                        filtered_candidates[person_id] = candidate_app
                        print(f"[Appearance] Candidate {person_id} matches: desc='{candidate_shirt_desc[:50]}...', color='{candidate_shirt_color}'")
                
                # If we have filtered candidates, use only those
                if len(filtered_candidates) > 0:
                    print(f"[Appearance] Filtered to {len(filtered_candidates)} candidates by shirt features")
                    candidates_to_score = filtered_candidates
                else:
                    # No shirt match, use all candidates (maybe extraction was wrong)
                    print(f"[Appearance] No shirt matches, using all {len(candidate_appearances)} candidates")
                    candidates_to_score = candidate_appearances
            else:
                # No shirt info extracted, use all candidates
                print(f"[Appearance] No shirt info extracted, using all {len(candidate_appearances)} candidates")
                candidates_to_score = candidate_appearances
            
            # Step 2: Score each candidate with full appearance matching
            best_match = None
            best_score = 0.0
            
            for person_id, candidate_app in candidates_to_score.items():
                score = self._score_appearance_match(leave_appearance, candidate_app)
                print(f"[Appearance] Match score for {person_id}: {score:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_match = person_id
            
            # Only return if score is above threshold
            if best_match and best_score > 0.5:  # 50% match threshold
                print(f"[Appearance] Best match: {best_match} with score {best_score:.2f}")
                return (best_match, best_score)
            else:
                print(f"[Appearance] No good match found (best score: {best_score:.2f})")
                return None
                
        except Exception as e:
            print(f"[Appearance] Error matching appearance: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _log_appearance_extraction(self, appearance: Dict, image: np.ndarray):
        """Log appearance extraction to file"""
        try:
            log_dir = config.EXPORTS_DIR
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"appearance_log_{timestamp}.json"
            
            # Create log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "appearance": appearance,
                "image_shape": list(image.shape),
                "model": self.model
            }
            
            # Append to log file (or create new)
            log_data = []
            if log_file.exists():
                with open(log_file, 'r') as f:
                    try:
                        log_data = json.load(f)
                        if not isinstance(log_data, list):
                            log_data = [log_data]
                    except json.JSONDecodeError:
                        log_data = []
            
            log_data.append(log_entry)
            
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            print(f"[Appearance] Logged extraction to {log_file}")
        except Exception as e:
            print(f"[Appearance] Failed to log extraction: {e}")
    
    def _score_appearance_match(self, appearance1: Dict, appearance2: Dict) -> float:
        """Score how well two appearance descriptions match
        
        Prioritizes distinctive features (shirt details, patterns) over colors
        which can be affected by lighting.
        
        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0
        total_weight = 0.0
        
        # Shirt description is most reliable (weight: 0.35) - includes patterns, logos, text
        shirt_desc1 = appearance1.get("shirt_description", "").lower()
        shirt_desc2 = appearance2.get("shirt_description", "").lower()
        if shirt_desc1 and shirt_desc2:
            # Check for distinctive features (logos, text, patterns)
            distinctive_words = ["logo", "text", "pattern", "stripes", "design", "print", "logo"]
            desc1_has_distinctive = any(word in shirt_desc1 for word in distinctive_words)
            desc2_has_distinctive = any(word in shirt_desc2 for word in distinctive_words)
            
            if desc1_has_distinctive and desc2_has_distinctive:
                # Both have distinctive features - check if they match
                # Extract key distinctive words
                desc1_words = set(word for word in distinctive_words if word in shirt_desc1)
                desc2_words = set(word for word in distinctive_words if word in shirt_desc2)
                if desc1_words.intersection(desc2_words):
                    score += 0.35  # Strong match on distinctive features
                else:
                    score += 0.15  # Both have distinctive features but different
            elif shirt_desc1 == shirt_desc2:
                score += 0.35  # Exact match
            else:
                # Check for partial match (e.g., both mention "white text" or similar patterns)
                desc1_words = set(shirt_desc1.split())
                desc2_words = set(shirt_desc2.split())
                common_words = desc1_words.intersection(desc2_words)
                if len(common_words) >= 2:  # At least 2 common words
                    similarity = len(common_words) / max(len(desc1_words), len(desc2_words))
                    score += 0.35 * similarity
                else:
                    # Fallback to color match
                    shirt_color1 = appearance1.get("shirt_color", "").lower()
                    shirt_color2 = appearance2.get("shirt_color", "").lower()
                    if shirt_color1 and shirt_color2:
                        if shirt_color1 == shirt_color2:
                            score += 0.2
                        elif shirt_color1 in shirt_color2 or shirt_color2 in shirt_color1:
                            score += 0.1
            total_weight += 0.35
        
        # Pants description (weight: 0.15) - only if visible in both
        pants_desc1 = appearance1.get("pants_description", "").lower()
        pants_desc2 = appearance2.get("pants_description", "").lower()
        if pants_desc1 and pants_desc2 and "not in picture" not in pants_desc1 and "not in picture" not in pants_desc2:
            if pants_desc1 == pants_desc2:
                score += 0.15
            else:
                # Partial match on pants
                pants_words1 = set(pants_desc1.split())
                pants_words2 = set(pants_desc2.split())
                common_pants = pants_words1.intersection(pants_words2)
                if len(common_pants) >= 1:
                    similarity = len(common_pants) / max(len(pants_words1), len(pants_words2))
                    score += 0.15 * similarity
            total_weight += 0.15
        
        # Height estimate (weight: 0.2)
        if appearance1.get("height_estimate") and appearance2.get("height_estimate"):
            if appearance1["height_estimate"].lower() == appearance2["height_estimate"].lower():
                score += 0.2
            total_weight += 0.2
        
        # Hair color (weight: 0.1) - may be visible from back, but lighting affects this
        if appearance1.get("hair_color") and appearance2.get("hair_color"):
            hair1 = appearance1["hair_color"].lower()
            hair2 = appearance2["hair_color"].lower()
            if hair1 == hair2:
                score += 0.1
            # Allow for lighting variations (e.g., "dirty blonde" vs "blonde")
            elif "blonde" in hair1 and "blonde" in hair2:
                score += 0.05  # Partial match
            total_weight += 0.1
        
        # Hair length (weight: 0.1)
        if appearance1.get("hair_length") and appearance2.get("hair_length"):
            if appearance1["hair_length"].lower() == appearance2["hair_length"].lower():
                score += 0.1
            total_weight += 0.1
        
        # Gender (weight: 0.05)
        if appearance1.get("gender") and appearance2.get("gender"):
            if appearance1["gender"].lower() == appearance2["gender"].lower():
                score += 0.05
            total_weight += 0.05
        
        # Description similarity (weight: 0.05) - use simple keyword matching
        if appearance1.get("description") and appearance2.get("description"):
            desc1_words = set(appearance1["description"].lower().split())
            desc2_words = set(appearance2["description"].lower().split())
            common_words = desc1_words.intersection(desc2_words)
            if len(desc1_words) > 0:
                similarity = len(common_words) / max(len(desc1_words), len(desc2_words))
                score += 0.05 * similarity
            total_weight += 0.05
        
        # Normalize score
        if total_weight > 0:
            return score / total_weight
        return 0.0

# Global instance
appearance_service = AppearanceService()

