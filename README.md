# Office Map API Server Setup Guide

## 🏗️ Architecture Overview

**You are here: The "Brain" (Server)**

```
[RPi5 Camera] --detects person with YOLO--> [RPi5 tracks & buffers images]
                                                        |
                                                        | HTTP POST
                                                        v
                                            [MacBook Server (YOU)]
                                            - Runs ArcFace (heavy AI)
                                            - Face recognition
                                            - State management
                                                        |
                                                        v
                                            [Web Browser polls /state]
                                            - Displays office map
```

## 📋 Prerequisites

- Python 3.8+
- Downloaded models:
  - `arcfaceresnet100-8.onnx` (~264 MB)
  - `haarcascade_frontalface_default.xml`

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Directory Structure

Your project should look like this:

```
office_map_project/
├── api_server.py
├── office_map.html
├── requirements.txt
├── models/
│   ├── arcfaceresnet100-8.onnx
│   └── haarcascade_frontalface_default.xml
└── static/
    └── profile_pics/
        └── (your avatar images, e.g., alice_smith.png)
```

### 3. Download Models

**ArcFace Model:**
```bash
# Download from ONNX Model Zoo
wget https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx -P models/
```

**Haar Cascade (OpenCV):**
```bash
# Download from OpenCV GitHub
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml -P models/
```

### 4. Run the Server

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
🧠 Loading ArcFace model...
✅ ArcFace model loaded
🧠 Loading Haar Cascade...
✅ Haar Cascade loaded
🚀 Server ready!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 5. Test the Server

Open your browser and go to: `http://localhost:8000`

You should see your office map HTML page.

## 🧪 Testing Without RPi5

You can test the server independently using `curl` or Python scripts before connecting your RPi5.

### Test 1: Register a Member (Simulated)

Create a simple test script to register yourself:

```python
# test_register.py
import requests
import cv2

# Capture 10 frames from your webcam
cap = cv2.VideoCapture(0)
images = []

print("Look at the camera...")
for i in range(10):
    ret, frame = cap.read()
    if ret:
        _, buffer = cv2.imencode('.jpg', frame)
        images.append(('images', ('frame.jpg', buffer.tobytes(), 'image/jpeg')))
    
cap.release()

# Send to server
response = requests.post(
    'http://localhost:8000/register',
    data={'member_id': 'test_user'},
    files=images
)

print(response.json())
```

Run it:
```bash
python test_register.py
```

Expected output:
```json
{
  "status": "success",
  "member_id": "test_user",
  "message": "Registered test_user with 10 images",
  "total_members": 1
}
```

### Test 2: Check State Endpoint

```bash
curl http://localhost:8000/state
```

Expected output:
```json
{
  "state": {
    "test_user": "out"
  },
  "total_members": 1,
  "members_present": 0
}
```

### Test 3: Debug Members

```bash
curl http://localhost:8000/debug/members
```

## 🔌 Connecting Your RPi5

Once your server is running, update your RPi5's `edge_tracker.py`:

```python
# In edge_tracker.py on your RPi5
BACKEND_API_URL = "http://192.168.1.100:8000"  # Replace with your MacBook's IP
```

**Find your MacBook's IP:**
```bash
# On macOS
ifconfig | grep "inet " | grep -v 127.0.0.1

# Or use:
ipconfig getifaddr en0  # For WiFi
```

## 📊 API Endpoints Reference

### `POST /register`
Register a new member with face images.

**Form Data:**
- `member_id` (string): Unique identifier (e.g., "alice_smith")
- `images` (files): 10 JPEG images

**Response:**
```json
{
  "status": "success",
  "member_id": "alice_smith",
  "message": "Registered alice_smith with 10 images",
  "total_members": 5
}
```

### `POST /event`
Handle entry/exit event.

**Form Data:**
- `direction` (string): "enter" or "leave"
- `images` (files): 10 JPEG images

**Response (Success):**
```json
{
  "status": "success",
  "member_id": "alice_smith",
  "similarity": 0.847,
  "direction": "enter",
  "new_state": "in",
  "old_state": "out"
}
```

**Response (No Match):**
```json
{
  "status": "no_match",
  "message": "Person not recognized",
  "direction": "enter"
}
```

### `GET /state`
Get current room occupancy (polled by frontend).

**Response:**
```json
{
  "state": {
    "alice_smith": "in",
    "bob_jones": "out",
    "charlie_kim": "in"
  },
  "total_members": 3,
  "members_present": 2
}
```

### `GET /debug/members`
Debug endpoint to see all registered members.

**Response:**
```json
{
  "registered_members": ["alice_smith", "bob_jones", "charlie_kim"],
  "state": {
    "alice_smith": "in",
    "bob_jones": "out",
    "charlie_kim": "in"
  }
}
```

## 🔧 Configuration & Tuning

### Face Recognition Threshold

In `api_server.py`, adjust this value:

```python
SIMILARITY_THRESHOLD = 0.5  # Default: 0.5
```

- **Too many false positives?** Increase to `0.6` or `0.7`
- **Not recognizing people?** Decrease to `0.4` or `0.45`
- Typical range: `0.3` to `0.7`

### Minimum Face Size

```python
MIN_FACE_SIZE = (30, 30)  # Default: 30x30 pixels
```

Increase if detecting too many false faces in the background.

## 🐛 Troubleshooting

### Problem: "ArcFace model not found"

**Solution:** Ensure `arcfaceresnet100-8.onnx` is in the `models/` directory.

```bash
ls -lh models/
# Should show arcfaceresnet100-8.onnx (~264 MB)
```

### Problem: "No face detected in any images"

**Causes:**
- Face is too small in the frame
- Poor lighting
- Face is at extreme angle
- Images are blurry

**Solutions:**
1. Have person stand closer to camera
2. Improve lighting
3. Increase buffer size from 10 to 20 images
4. Lower `MIN_FACE_SIZE` threshold

### Problem: "Person not recognized"

**Causes:**
- Similarity score below threshold
- Person not registered
- Very different lighting/angle from registration

**Debug Steps:**

1. Check if person is registered:
   ```bash
   curl http://localhost:8000/debug/members
   ```

2. Check similarity scores in server logs:
   ```
   🔍 Searching for match in database...
     • alice_smith: similarity = 0.823
     • bob_jones: similarity = 0.234
   ✅ Match found: alice_smith (similarity: 0.823)
   ```

3. If similarity is close to threshold (e.g., 0.48 when threshold is 0.5):
   - Lower `SIMILARITY_THRESHOLD`
   - Re-register with better quality images

### Problem: Server crashes on ARM Mac (M1/M2/M3)

**Cause:** ONNX Runtime may need ARM-specific configuration.

**Solution:**
```bash
pip uninstall onnxruntime
pip install onnxruntime  # This should auto-detect ARM
```

Or try:
```bash
pip install onnxruntime-silicon  # For Apple Silicon
```

## 📈 Performance Tips

### Current Performance (M-series Mac)
- Model loading: ~2-3 seconds
- Single image processing: ~50-100ms
- 10-image batch: ~500ms-1s

### Optimization Ideas

1. **GPU Acceleration** (if you have CUDA-capable GPU):
   ```python
   arcface_session = ort.InferenceSession(
       str(ARCFACE_MODEL_PATH),
       providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
   )
   ```

2. **Reduce Image Size** on RPi5:
   - Current: sending full-resolution crops
   - Optimization: resize to 640x480 before sending

3. **Async Processing**:
   - Current: blocking requests
   - Future: process images in background thread

## 🚢 Next Steps: Moving to Production

### 1. Replace FAKE_VECTOR_DB

**Option A: Vector Database**
- Pinecone (easiest, cloud-hosted)
- Weaviate (self-hosted)
- pgvector (PostgreSQL extension)

**Option B: Simple Persistence**
```python
import pickle

# Save on shutdown
with open('vector_db.pkl', 'wb') as f:
    pickle.dump(FAKE_VECTOR_DB, f)

# Load on startup
with open('vector_db.pkl', 'rb') as f:
    FAKE_VECTOR_DB = pickle.load(f)
```

### 2. Replace FAKE_STATE_DB with Real-Time DB

**Firebase Firestore** (Recommended):
```python
import firebase_admin
from firebase_admin import firestore

db = firestore.client()

# Update state
db.collection('office_state').document(member_id).set({
    'state': 'in',
    'timestamp': firestore.SERVER_TIMESTAMP
})
```

Frontend subscribes to changes via WebSocket (no polling needed).

### 3. Deploy Server

**Option A: Fly.io** (Simple)
```bash
flyctl launch
flyctl deploy
```

**Option B: Railway**
- Push to GitHub
- Connect Railway to repo
- Auto-deploys on push

**Option C: Docker + Any Cloud**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4. Add Authentication

```python
from fastapi import Depends, HTTPException, Header

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != "your-secret-key":
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key

@app.post("/event", dependencies=[Depends(verify_api_key)])
async def handle_event(...):
    ...
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [ArcFace Paper](https://arxiv.org/abs/1801.07698)
- [OpenCV Face Detection](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)

## 🆘 Need Help?

Check the server logs for detailed error messages:
```bash
# Server logs show:
# - Which images had faces detected
# - Similarity scores for each member
# - State changes
```

Example log output:
```
📝 Registration request for: alice_smith
   Received 10 images
  ✓ Image 1: Face detected and processed
  ✓ Image 2: Face detected and processed
  ...
✅ Generated mean embedding from 10/10 images
✅ Successfully registered alice_smith
   Total registered members: 1
```
