#!/bin/bash
# Test Docker build and run locally (matching Railway configuration)

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Testing Docker Build Locally"
echo "=========================================="
echo ""

# Build the Docker image
# Use project root as context (matching Railway's dockerContext: ".")
echo "Building Docker image..."
docker build \
    -f server/Dockerfile \
    -t office-map-server:test \
    .

echo ""
echo "=========================================="
echo "  Build Complete!"
echo "=========================================="
echo ""

# Run the container
echo "Starting container..."
echo "Server will be available at: http://localhost:8000"
echo "Press CTRL+C to stop"
echo ""

docker run --rm -it \
    -p 8000:8000 \
    -e PORT=8000 \
    office-map-server:test

