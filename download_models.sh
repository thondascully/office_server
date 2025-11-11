#!/bin/bash
# Download models for Office Map API Server

set -e  # Exit on error

echo "======================================================================"
echo "  Office Map API - Model Download Script"
echo "======================================================================"
echo ""

# Create models directory
mkdir -p models
echo "📁 Model directory: $(pwd)/server/models"
echo ""

# Download ArcFace model
echo "----------------------------------------------------------------------"
echo "📥 Downloading ArcFace Model (249 MB) - this may take a few minutes..."
echo "----------------------------------------------------------------------"

if [ -f "server/models/arcfaceresnet100-8.onnx" ]; then
    SIZE=$(du -h "server/models/arcfaceresnet100-8.onnx" | cut -f1)
    echo "⚠️  File already exists: server/models/arcfaceresnet100-8.onnx ($SIZE)"
    read -p "   Re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Skipping ArcFace download"
    else
        rm "server/models/arcfaceresnet100-8.onnx"
        curl -L --progress-bar \
            -o "server/models/arcfaceresnet100-8.onnx" \
            "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx"
        echo "✅ ArcFace model downloaded"
    fi
else
    curl -L --progress-bar \
        -o "server/models/arcfaceresnet100-8.onnx" \
        "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx"
    echo "✅ ArcFace model downloaded"
fi

echo ""

# Download Haar Cascade
echo "----------------------------------------------------------------------"
echo "📥 Downloading Haar Cascade (1 MB)..."
echo "----------------------------------------------------------------------"

if [ -f "server/models/haarcascade_frontalface_default.xml" ]; then
    SIZE=$(du -h "server/models/haarcascade_frontalface_default.xml" | cut -f1)
    echo "⚠️  File already exists: server/models/haarcascade_frontalface_default.xml ($SIZE)"
    read -p "   Re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Skipping Haar Cascade download"
    else
        rm "server/models/haarcascade_frontalface_default.xml"
        curl -L --progress-bar \
            -o "server/models/haarcascade_frontalface_default.xml" \
            "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        echo "✅ Haar Cascade downloaded"
    fi
else
    curl -L --progress-bar \
        -o "server/models/haarcascade_frontalface_default.xml" \
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    echo "✅ Haar Cascade downloaded"
fi

echo ""
echo "======================================================================"
echo "  Download Summary"
echo "======================================================================"

# Check files
if [ -f "server/models/arcfaceresnet100-8.onnx" ]; then
    SIZE=$(du -h "server/models/arcfaceresnet100-8.onnx" | cut -f1)
    echo "✅ arcfaceresnet100-8.onnx ($SIZE)"
else
    echo "❌ arcfaceresnet100-8.onnx - MISSING"
fi

if [ -f "server/models/haarcascade_frontalface_default.xml" ]; then
    SIZE=$(du -h "server/models/haarcascade_frontalface_default.xml" | cut -f1)
    echo "✅ haarcascade_frontalface_default.xml ($SIZE)"
else
    echo "❌ haarcascade_frontalface_default.xml - MISSING"
fi

echo "======================================================================"
echo ""

# Final check
if [ -f "server/models/arcfaceresnet100-8.onnx" ] && [ -f "server/models/haarcascade_frontalface_default.xml" ]; then
    echo "🎉 All models downloaded successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Install dependencies: pip install -r requirements.txt"
    echo "  2. Run server: python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload"
    echo ""
else
    echo "⚠️  Some models failed to download."
    echo ""
    echo "Try downloading manually:"
    echo ""
    echo "1. ArcFace Model:"
    echo "   curl -L -o server/models/arcfaceresnet100-8.onnx \\"
    echo "     'https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx'"
    echo ""
    echo "2. Haar Cascade:"
    echo "   curl -o server/models/haarcascade_frontalface_default.xml \\"
    echo "     'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'"
    echo ""
fi
