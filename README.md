# Office Map System - Complete Documentation

## System Overview

The Office Map System is a distributed face recognition and presence tracking system consisting of:
- **Raspberry Pi 5 (Edge Device)**: Motion detection, person detection, and image capture
- **Central Server**: Face recognition, state management, and data storage
- **Web Dashboard**: Real-time monitoring and management interface

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Raspberry Pi 5 (Edge Device)                  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Motion Detection Loop (Every 1 second)                  │  │
│  │  - MAE Randomized Algorithm                               │  │
│  │  - Detects frame-to-frame motion changes                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           │ Motion Detected                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Tripwire Filter                                         │  │
│  │  - Check if motion is within tripwire boundaries         │  │
│  │  - Outer X: Entrance boundary (default: 800px)          │  │
│  │  - Inner X: Office boundary (default: 1800px)            │  │
│  │  - Only process if motion within these bounds            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           │ Within Tripwires                     │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  YOLO Person Detection                                   │  │
│  │  - Runs YOLO model to detect if motion is a person       │  │
│  │  - Filters out false positives (pets, objects, etc.)     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           │ Person Detected                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Image Capture & Buffering                               │  │
│  │  - Captures 10 images in burst                           │  │
│  │  - Buffers images for transmission                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           │ HTTP POST /api/event                 │
│                           ▼                                      │
└─────────────────────────────────────────────────────────────────┘
                           │
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Central Server (FastAPI)                      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Event Handler (/api/event)                              │  │
│  │  - Receives 10 images from RPi                          │  │
│  │  - Processes up to 5 images (optimized for speed)       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Face Detection (Haar Cascade)                           │  │
│  │  - Detects faces in images                                │  │
│  │  - Multi-strategy fallback for reliability                │  │
│  │  - Resizes images to 1600px max width for speed           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           │ Face Detected                        │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Face Recognition (ArcFace)                              │  │
│  │  - Generates 512-dimensional embedding                   │  │
│  │  - Computes mean embedding from multiple images           │  │
│  │  - Early stopping after 3-4 good embeddings              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Vector Database Search                                   │  │
│  │  - Cosine similarity matching                            │  │
│  │  - Threshold: 0.5 (configurable)                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│              ┌────────────┴────────────┐                        │
│              │                         │                         │
│         Match Found              No Match                         │
│              │                         │                         │
│              ▼                         ▼                         │
│  ┌──────────────────┐    ┌──────────────────────────────┐        │
│  │  Update State    │    │  Auto-Register Unknown      │        │
│  │  - Set in/out    │    │  - Create new person_id     │        │
│  │  - Update times  │    │  - Store embedding          │        │
│  │                  │    │  - Save sample image         │        │
│  └──────────────────┘    └──────────────────────────────┘        │
│              │                         │                         │
│              └────────────┬────────────┘                         │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Metadata Database                                        │  │
│  │  - Store person information                               │  │
│  │  - Track state changes                                    │  │
│  │  - Save to metadata.json                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           │                                      │
│                           ▼                                      │
└─────────────────────────────────────────────────────────────────┘
                           │
                           │ HTTP GET /api/dashboard/*
                           │ (Polled every 10 seconds)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Web Dashboard (Browser)                      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Auto-Refresh Loop                                        │  │
│  │  - Fetches stats, people, RPi status                     │  │
│  │  - Updates UI smoothly without flickering                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Display Components                                       │  │
│  │  - Currently in office                                    │  │
│  │  - Recent activity feed                                   │  │
│  │  - People database                                        │  │
│  │  - Unlabeled people                                       │  │
│  │  - RPi connection status                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Workflow

### Raspberry Pi 5 Edge Device

#### Motion Detection (MAE Randomized Algorithm)
- **Frequency**: Checks every 1 second
- **Method**: MAE (Mean Absolute Error) randomized algorithm compares consecutive frames
- **Purpose**: Efficiently detects any motion in the camera view
- **Optimization**: Only triggers expensive operations when motion is detected

#### Tripwire Filtering
- **Purpose**: Reduces false positives and saves computation
- **Configuration**:
  - `outer_x`: Entrance boundary (default: 800px from left)
  - `inner_x`: Office boundary (default: 1800px from left)
- **Logic**: Only processes detections if motion/person is within these X-axis boundaries
- **Benefits**:
  - Prevents processing people walking by outside the office
  - Reduces unnecessary YOLO runs
  - Saves computational resources on RPi

#### Person Detection (YOLO)
- **Trigger**: Only runs when motion is detected AND within tripwires
- **Model**: YOLO object detection model
- **Purpose**: Confirms that detected motion is actually a person
- **Filtering**: Eliminates false positives (pets, moving objects, shadows)
- **Output**: Person bounding box coordinates

#### Image Capture
- **Burst Size**: 10 images captured when person detected
- **Timing**: Images captured with small delays between frames
- **Format**: JPEG images ready for transmission
- **Transmission**: Sent via HTTP POST to server

#### Communication with Server
- **Heartbeat**: Periodic status updates to `/api/rpi/heartbeat`
- **Command Polling**: Checks `/api/rpi/commands/{rpi_id}` for dashboard commands
- **Stream Frames**: Sends live camera feed to `/api/rpi/stream/{rpi_id}`
- **Event Submission**: POSTs detection events to `/api/event`

### Central Server

#### Face Recognition Pipeline

**1. Image Reception**
- Receives up to 10 images from RPi
- Processes first 5 images (optimized for speed)
- Handles image decoding errors gracefully

**2. Face Detection (Haar Cascade)**
- Uses OpenCV Haar Cascade classifier
- Multi-strategy fallback:
  - Strategy 1: Lenient parameters (fast, catches most faces)
  - Strategy 2: Smaller minimum size if no faces found
  - Strategy 3: Very lenient last resort
- Image preprocessing: Resizes to max 1600px width before detection
- Output: Face bounding box coordinates

**3. Face Embedding (ArcFace)**
- Model: `arcfaceresnet100-8.onnx` (ONNX Runtime)
- Input: 112x112 cropped and aligned face images
- Output: 512-dimensional embedding vector
- Optimization: Early stopping after 3-4 good embeddings
- Mean embedding: Averages multiple face embeddings for robustness

**4. Vector Database Search**
- Method: Cosine similarity matching
- Threshold: 0.5 (configurable in `config.py`)
- Process:
  1. Normalize query embedding
  2. Compare against all stored embeddings
  3. Return best match if similarity >= threshold
  4. Return None if no match found

**5. State Management**
- **Known Person**:
  - Updates existing person's state (in/out)
  - Records entry/exit timestamps
  - Tracks visit count
  - Stores recognition confidence
- **Unknown Person**:
  - Auto-registers with generated ID (e.g., `unknown_012`)
  - Creates metadata entry
  - Stores face embedding
  - Saves sample image
  - Can be labeled later via dashboard

#### Data Storage

**Vector Database** (`vectors.json`)
- Stores 512-dimensional face embeddings
- Key: `person_id`
- Value: List of floats (embedding vector)
- Format: JSON file

**Metadata Database** (`metadata.json`)
- Stores person information:
  - `person_id`: Unique identifier
  - `name`: Display name (optional)
  - `state`: "in" or "out"
  - `created_at`: First registration timestamp
  - `last_seen`: Most recent detection timestamp
  - `entered_at`: When person entered (if currently in)
  - `last_exit`: When person last left (if currently out)
  - `visit_count`: Number of times person has entered
  - `last_similarity`: Recognition confidence score
  - `image_paths`: List of stored image file paths

### Web Dashboard

#### Real-Time Updates
- **Auto-refresh**: Polls server every 10 seconds
- **Endpoints**:
  - `/api/dashboard/stats`: Statistics (total people, present, unlabeled)
  - `/api/dashboard/people`: All people with metadata
  - `/api/rpi/status`: RPi connection status
- **Smooth Updates**: Uses data attributes to update DOM without flickering

#### Features
- **Currently In Office**: Shows people currently present
- **Recent Activity**: Timeline of entry/exit events
- **People Database**: Browse and manage all registered people
- **Unlabeled People**: View and label unknown persons
- **RPi Management**: View live stream, register people, calibrate tripwires
- **Database Management**: Backup/restore full state or vectors only

## System Flow Example

### Scenario: Person Enters Office

1. **RPi Motion Detection** (Every 1s)
   - MAE algorithm detects motion in frame
   - Motion detected at X=1200px (within tripwires 800-1800)

2. **Tripwire Check**
   - X=1200px is between outer_x (800) and inner_x (1800)
   - Motion is within valid zone
   - Proceed to person detection

3. **YOLO Person Detection**
   - YOLO model processes frame
   - Person detected with confidence > threshold
   - Person bounding box confirmed

4. **Image Capture**
   - RPi captures 10 images in burst
   - Images buffered for transmission

5. **Event Transmission**
   - RPi sends POST to `/api/event`
   - Includes: direction="enter", 10 images, timestamp

6. **Server Processing**
   - Receives 10 images, processes first 5
   - Face detection finds face in 4/5 images
   - ArcFace generates embeddings from 4 faces
   - Mean embedding computed

7. **Vector Search**
   - Compares mean embedding against database
   - Best match: "alice_smith" with similarity 0.87
   - Similarity > threshold (0.5), match confirmed

8. **State Update**
   - Updates alice_smith state: "out" -> "in"
   - Records entered_at timestamp
   - Increments visit_count
   - Stores last_similarity (0.87)
   - Saves to metadata.json

9. **Dashboard Update**
   - Dashboard polls `/api/dashboard/people`
   - Receives updated state for alice_smith
   - UI updates "Currently In Office" section
   - Activity feed shows "Alice Smith entered"

### Scenario: Unknown Person Enters

1-5. Same as above (motion detection through image capture)

6. **Server Processing**
   - Face detection and embedding generation (same as above)

7. **Vector Search**
   - Compares embedding against database
   - Best match: similarity 0.35 (below threshold 0.5)
   - No match found

8. **Auto-Registration**
   - Creates new person_id: "unknown_012"
   - Stores face embedding in vector database
   - Creates metadata entry (name=None, state="in")
   - Saves sample image to disk
   - Returns status: "unknown_registered"

9. **Dashboard Update**
   - Dashboard shows "unknown_012" in unlabeled section
   - Admin can click to label the person
   - Once labeled, person appears in main database

## Configuration

### Server Configuration (`server/config.py`)

```python
# Face Recognition
SIMILARITY_THRESHOLD = 0.5  # Matching threshold
MIN_FACE_SIZE = (30, 30)    # Minimum face size in pixels
DETECTION_MAX_WIDTH = 1600  # Max image width for detection (speed optimization)

# Image Storage
MAX_IMAGES_PER_PERSON = 20  # Maximum stored images per person
THUMBNAIL_SIZE = (150, 150) # Thumbnail dimensions
```

### RPi Configuration (`server/data/rpi_configs/default.yaml`)

```yaml
burst_size: 10              # Number of images to capture
detection_interval: 1.0      # Motion detection interval (seconds)
tripwires:
  inner_x: 1800              # Inner boundary (office)
  outer_x: 800               # Outer boundary (entrance)
```

## API Endpoints

### Face Recognition

**POST /api/register**
- Register a new person with face images
- Parameters: `name` (optional), `rpi_id`, `images` (files)
- Returns: Registration status and person_id

**POST /api/event**
- Handle entry/exit event from RPi
- Parameters: `direction` ("enter"/"leave"), `rpi_id`, `timestamp`, `images` (files)
- Returns: Recognition result, person_id, similarity score, state change

### Dashboard

**GET /**
- Main dashboard page (HTML)

**GET /api/dashboard/stats**
- Get dashboard statistics
- Returns: Total people, present count, unlabeled count, vector count, connected RPis

**GET /api/dashboard/people**
- Get all registered people with metadata
- Returns: Array of person objects with full details

**POST /api/dashboard/label**
- Label an unlabeled person
- Body: `person_id`, `name`
- Returns: Success status

**DELETE /api/dashboard/person/{person_id}**
- Delete a person from the system
- Returns: Success status

### RPi Communication

**POST /api/rpi/heartbeat**
- RPi sends periodic heartbeat
- Body: `rpi_id`, `status`, `uptime`
- Returns: Server time

**GET /api/rpi/commands/{rpi_id}**
- RPi polls for commands from dashboard
- Returns: Command and parameters (if any)

**GET /api/rpi/config/{rpi_id}**
- Get RPi configuration
- Returns: Configuration object including tripwires

**POST /api/rpi/config/{rpi_id}**
- Update RPi configuration
- Body: Configuration object
- Returns: Success status

**POST /api/rpi/stream/{rpi_id}**
- Receive stream frame from RPi
- Body: JPEG image bytes
- Returns: Status

**GET /api/rpi/stream/{rpi_id}**
- Get latest stream frame for dashboard
- Returns: JPEG image

**GET /api/rpi/status**
- Get status of all connected RPis
- Returns: Array of RPi status objects

### RPi Control

**POST /api/rpi/system-toggle**
- Enable/disable RPi system
- Body: `rpi_id`, `enabled`
- Returns: Success status

**POST /api/rpi/start-stream**
- Command RPi to start streaming
- Body: `rpi_id`
- Returns: Success status

**POST /api/rpi/stop-stream**
- Command RPi to stop streaming
- Body: `rpi_id`
- Returns: Success status

**POST /api/rpi/calibrate**
- Command RPi to enter calibration mode
- Body: `rpi_id`
- Returns: Success status

**POST /api/rpi/stop-calibrate**
- Command RPi to exit calibration mode
- Body: `rpi_id`
- Returns: Success status

### Database Management

**GET /api/export/vectors**
- Export vector database only
- Returns: JSON file download

**POST /api/import/vectors**
- Import vector database
- Body: JSON file
- Returns: Import status and count

**GET /api/export/full-state**
- Export complete system state (vectors + metadata)
- Returns: JSON file download

**POST /api/import/full-state**
- Import complete system state
- Body: JSON file
- Returns: Import status and counts

## Installation & Setup

### Prerequisites

- Python 3.8+
- Raspberry Pi 5 with camera module
- Downloaded models:
  - `arcfaceresnet100-8.onnx` (~264 MB)
  - `haarcascade_frontalface_default.xml`

### Server Setup

1. **Install Dependencies**
```bash
cd server
pip install -r requirements.txt
```

2. **Download Models**
```bash
# Use the download script from project root
./scripts/download_models.sh

# Or manually download to server/models/:
# ArcFace model
wget https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx -P server/models/

# Haar Cascade
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml -P server/models/
```

3. **Run Server**
```bash
# From server directory
python main.py

# Or use the startup script from project root
./scripts/start_server.sh
```

Server will start on `http://0.0.0.0:8000`

### RPi Setup

1. **Configure Server URL**
   - Update RPi code to point to server IP
   - Example: `http://192.168.1.100:8000`

2. **Calibrate Tripwires**
   - Use dashboard "Calibrate Tripwires" feature
   - Drag lines to set outer and inner boundaries
   - Save configuration

3. **Start RPi Detection**
   - RPi begins motion detection loop
   - Sends heartbeat every few seconds
   - Processes detections and sends events

## Deployment

### Railway Deployment

The project includes Railway configuration:

- **railway.json**: Deployment configuration
- **Dockerfile**: Container build instructions
- **Health Check**: `/health` endpoint for Railway monitoring

Deploy by connecting Railway to your GitHub repository.

### Local Docker Testing

```bash
# Build and test locally
./scripts/test_docker_local.sh
```

## Troubleshooting

### Server Issues

**Problem: "ArcFace model not found"**
- Ensure `arcfaceresnet100-8.onnx` is in `server/models/`
- Check file permissions

**Problem: "No faces detected in any images"**
- Check lighting conditions
- Verify person is facing camera
- Adjust `MIN_FACE_SIZE` if faces are too small
- Check `DETECTION_MAX_WIDTH` setting

**Problem: Person not recognized**
- Check similarity threshold in `config.py`
- Verify person is registered
- Re-register with better quality images
- Check server logs for similarity scores

### RPi Connection Issues

**Problem: RPi not connecting**
- Verify server URL in RPi configuration
- Check network connectivity
- Verify server is accessible from RPi network
- Check firewall settings

**Problem: Events timing out**
- Increase server timeout settings
- Reduce number of images sent (currently 10)
- Check network latency
- Verify server processing speed

### Dashboard Issues

**Problem: Dashboard not loading**
- Check browser console for errors
- Verify API endpoints are accessible
- Check CORS settings
- Verify password is correct (ends with "000")

## Performance Optimizations

### Server-Side

- **Image Processing**: Limited to 5 images per event (configurable)
- **Face Detection**: Image resizing to 1600px max width before detection
- **Early Stopping**: Stops after 3-4 good embeddings for speed
- **Multi-strategy Detection**: Fallback strategies for reliability

### RPi-Side

- **Motion Detection**: MAE algorithm is lightweight (runs every 1s)
- **Tripwire Filtering**: Prevents unnecessary YOLO runs
- **Burst Capture**: 10 images captured efficiently
- **Image Compression**: JPEG format for transmission

## Data Management

### Backup & Restore

**Full State Backup**
- Exports both vectors and metadata
- Use "Download Full State" in dashboard
- Restore with "Upload Full State"
- Useful for deployment migrations

**Vector Database Only**
- Exports only face embeddings
- Use "Download Vector DB" in dashboard
- Restore with "Upload Vector DB"
- Useful for sharing recognition data

### Data Files

- `server/data/vectors.json`: Face embeddings database
- `server/data/metadata.json`: Person metadata database
- `server/data/rpi_configs/default.yaml`: RPi configuration
- `server/static/images/`: Stored person images

## Security Considerations

- Dashboard is password protected (password must end with "000")
- All timestamps stored in UTC
- Images stored locally on server
- No external data sharing by default
- CORS enabled for development (restrict in production)

## Future Enhancements

- WebSocket support for real-time updates (replacing polling)
- Multiple RPi device support with location tracking
- Advanced analytics and reporting
- Mobile app for member self-service
- Privacy controls and opt-in/opt-out features
- Automated data retention policies
