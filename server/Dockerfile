# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY server/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire server directory to /app
COPY server/ .

# Create necessary directories if they don't exist
RUN mkdir -p data static/images static/exports models

# Download required models
RUN echo "Downloading face recognition models..." && \
    curl -L --progress-bar \
        -o models/arcfaceresnet100-8.onnx \
        "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx" && \
    curl -L --progress-bar \
        -o models/haarcascade_frontalface_default.xml \
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml" && \
    echo "Models downloaded successfully"

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Verify main.py exists (for debugging)
RUN ls -la /app/ | head -20 && \
    test -f /app/main.py && echo "main.py found" || echo "ERROR: main.py not found"

# Set PYTHONPATH environment variable
ENV PYTHONPATH=/app

# Run the server using uvicorn directly for better production performance
# Use PORT environment variable if set, otherwise default to 8000
# Add debugging to see what's in /app
CMD ["sh", "-c", "echo '=== Debugging container contents ===' && ls -la /app/ && echo '=== Checking for main.py ===' && test -f /app/main.py && echo 'main.py exists' || echo 'main.py NOT FOUND' && echo '=== Python path ===' && python -c 'import sys; print(sys.path)' && echo '=== Starting server ===' && cd /app && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

