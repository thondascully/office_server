#!/bin/bash
# Download models for Office Map API Server

# Don't exit on error - RetinaFace is optional and we want to continue if it fails
set +e

echo "======================================================================"
echo "  Office Map API - Model Download Script"
echo "======================================================================"
echo ""

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Create models directory
mkdir -p "$PROJECT_ROOT/server/models"
echo "📁 Model directory: $PROJECT_ROOT/server/models"
echo ""

# Download ArcFace model
echo "----------------------------------------------------------------------"
echo "📥 Downloading ArcFace Model (249 MB) - this may take a few minutes..."
echo "----------------------------------------------------------------------"

ARCFACE_URL="https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx"
ARCFACE_INT8_URL="https://huggingface.co/onnxmodelzoo/arcfaceresnet100-11-int8/resolve/main/model/arcfaceresnet100-11-int8.onnx"

if [ -f "$PROJECT_ROOT/server/models/arcfaceresnet100-8.onnx" ]; then
    SIZE=$(du -h "$PROJECT_ROOT/server/models/arcfaceresnet100-8.onnx" | cut -f1)
    echo "⚠️  File already exists: $PROJECT_ROOT/server/models/arcfaceresnet100-8.onnx ($SIZE)"
    read -p "   Re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Skipping ArcFace download"
    else
        rm "$PROJECT_ROOT/server/models/arcfaceresnet100-8.onnx"
        curl -L --progress-bar \
            -o "$PROJECT_ROOT/server/models/arcfaceresnet100-8.onnx" \
            "$ARCFACE_URL"
        echo "✅ ArcFace model downloaded"
    fi
else
    curl -L --progress-bar \
        -o "$PROJECT_ROOT/server/models/arcfaceresnet100-8.onnx" \
        "$ARCFACE_URL"
    echo "✅ ArcFace model downloaded"
fi

echo ""

# Download RetinaFace model (optional but recommended for better accuracy)
echo "----------------------------------------------------------------------"
echo "📥 Downloading RetinaFace Model (optional, ~1.7 MB)..."
echo "----------------------------------------------------------------------"
echo "   RetinaFace provides much better face detection than Haar Cascade"
echo "   It's more accurate and reduces false positives (like detecting arms as faces)"
echo ""

RETINAFACE_URL="https://huggingface.co/TheEeeeLin/HivisionIDPhotos_matting/resolve/main/retinaface-resnet50.onnx"
RETINAFACE_FILENAME="retinaface_r50_v1.onnx"

if [ -f "$PROJECT_ROOT/server/models/$RETINAFACE_FILENAME" ]; then
    SIZE=$(du -h "$PROJECT_ROOT/server/models/$RETINAFACE_FILENAME" | cut -f1)
    echo "⚠️  File already exists: $PROJECT_ROOT/server/models/$RETINAFACE_FILENAME ($SIZE)"
    read -p "   Re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Skipping RetinaFace download"
    else
        rm "$PROJECT_ROOT/server/models/$RETINAFACE_FILENAME"
        echo "   Downloading RetinaFace ResNet50 model from HuggingFace..."
        curl -L --progress-bar \
            -o "$PROJECT_ROOT/server/models/$RETINAFACE_FILENAME" \
            "$RETINAFACE_URL"
        if [ $? -eq 0 ] && [ -f "$PROJECT_ROOT/server/models/$RETINAFACE_FILENAME" ]; then
            echo "✅ RetinaFace model downloaded"
        else
            echo "❌ Failed to download RetinaFace model"
            echo "   The system will use Haar Cascade as fallback"
        fi
    fi
else
    echo "   Downloading RetinaFace ResNet50 model from HuggingFace..."
    curl -L --progress-bar \
        -o "$PROJECT_ROOT/server/models/$RETINAFACE_FILENAME" \
        "$RETINAFACE_URL"
    if [ $? -eq 0 ] && [ -f "$PROJECT_ROOT/server/models/$RETINAFACE_FILENAME" ]; then
        echo "✅ RetinaFace model downloaded"
    else
        echo "❌ Failed to download RetinaFace model"
        echo "   The system will use Haar Cascade as fallback"
    fi
fi

# Download UltraFace model (lightweight, fast face detector for tiling)
echo "----------------------------------------------------------------------"
echo "📥 Downloading UltraFace Model (~1.2 MB)..."
echo "----------------------------------------------------------------------"
echo "   UltraFace is a lightweight, fast face detector"
echo "   Used for tiling approach to detect faces in high-resolution images"
echo ""

# Try multiple UltraFace URLs (some may not be available)
ULTRAFACE_URL_640="https://github.com/onnx/models/raw/main/vision/body_analysis/ultraface/models/version-RFB-640.onnx"
ULTRAFACE_URL_320="https://github.com/onnx/models/raw/main/vision/body_analysis/ultraface/models/version-RFB-320.onnx"
ULTRAFACE_FILENAME="ultraface_rfb_640.onnx"

if [ -f "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" ]; then
    SIZE=$(du -h "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" | cut -f1)
    echo "⚠️  File already exists: $PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME ($SIZE)"
    read -p "   Re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Skipping UltraFace download"
    else
        rm "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME"
        echo "   Downloading UltraFace model from ONNX Model Zoo..."
        # Try 640 version first
        curl -L --progress-bar \
            -o "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" \
            "$ULTRAFACE_URL_640"
        
        # Check if download was successful and valid
        if [ -f "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" ]; then
            if head -1 "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" | grep -q "<!DOCTYPE html>" || [ ! -s "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" ]; then
                echo "   ⚠️  640 version not available, trying 320 version..."
                rm "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME"
                curl -L --progress-bar \
                    -o "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" \
                    "$ULTRAFACE_URL_320"
            fi
        fi
        
        if [ -f "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" ] && [ -s "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" ] && ! head -1 "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" | grep -q "<!DOCTYPE html>"; then
            SIZE=$(du -h "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" | cut -f1)
            echo "✅ UltraFace model downloaded ($SIZE)"
        else
            echo "❌ Failed to download UltraFace model"
            rm -f "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME"
        fi
    fi
else
    echo "   Downloading UltraFace RFB-640 model from ONNX Model Zoo..."
    # Try 640 version first
    curl -L --progress-bar \
        -o "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" \
        "$ULTRAFACE_URL_640"
    
    # Check if download was successful (file exists and is not HTML/empty)
    if [ -f "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" ]; then
        # Check if file is valid (not HTML error page)
        if head -1 "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" | grep -q "<!DOCTYPE html>" || [ ! -s "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" ]; then
            echo "   ⚠️  640 version not available, trying 320 version..."
            rm "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME"
            curl -L --progress-bar \
                -o "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" \
                "$ULTRAFACE_URL_320"
        fi
    fi
    
    # Final check
    if [ -f "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" ] && [ -s "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" ] && ! head -1 "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" | grep -q "<!DOCTYPE html>"; then
        SIZE=$(du -h "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME" | cut -f1)
        echo "✅ UltraFace model downloaded ($SIZE)"
    else
        echo "❌ Failed to download UltraFace model (URLs may be unavailable)"
        echo "   Tiling will use Haar Cascade fallback"
        rm -f "$PROJECT_ROOT/server/models/$ULTRAFACE_FILENAME"
    fi
fi

echo ""

# Download Haar Cascade (fallback)
echo "----------------------------------------------------------------------"
echo "📥 Downloading Haar Cascade (fallback, 1 MB)..."
echo "----------------------------------------------------------------------"

if [ -f "$PROJECT_ROOT/server/models/haarcascade_frontalface_default.xml" ]; then
    SIZE=$(du -h "$PROJECT_ROOT/server/models/haarcascade_frontalface_default.xml" | cut -f1)
    echo "⚠️  File already exists: $PROJECT_ROOT/server/models/haarcascade_frontalface_default.xml ($SIZE)"
    read -p "   Re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Skipping Haar Cascade download"
    else
        rm "$PROJECT_ROOT/server/models/haarcascade_frontalface_default.xml"
        curl -L --progress-bar \
            -o "$PROJECT_ROOT/server/models/haarcascade_frontalface_default.xml" \
            "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        echo "✅ Haar Cascade downloaded"
    fi
else
    curl -L --progress-bar \
        -o "$PROJECT_ROOT/server/models/haarcascade_frontalface_default.xml" \
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    echo "✅ Haar Cascade downloaded"
fi

echo ""
echo "======================================================================"
echo "  Download Summary"
echo "======================================================================"

# Check files
if [ -f "$PROJECT_ROOT/server/models/arcfaceresnet100-8.onnx" ]; then
    SIZE=$(du -h "$PROJECT_ROOT/server/models/arcfaceresnet100-8.onnx" | cut -f1)
    echo "✅ arcfaceresnet100-8.onnx ($SIZE)"
else
    echo "❌ arcfaceresnet100-8.onnx - MISSING"
fi

if [ -f "$PROJECT_ROOT/server/models/retinaface_r50_v1.onnx" ]; then
    SIZE=$(du -h "$PROJECT_ROOT/server/models/retinaface_r50_v1.onnx" | cut -f1)
    echo "✅ retinaface_r50_v1.onnx ($SIZE) - RetinaFace detector"
else
    echo "⚠️  retinaface_r50_v1.onnx - NOT FOUND (will use Haar Cascade fallback)"
fi

if [ -f "$PROJECT_ROOT/server/models/ultraface_rfb_640.onnx" ]; then
    SIZE=$(du -h "$PROJECT_ROOT/server/models/ultraface_rfb_640.onnx" | cut -f1)
    echo "✅ ultraface_rfb_640.onnx ($SIZE) - UltraFace detector (for tiling)"
else
    echo "⚠️  ultraface_rfb_640.onnx - NOT FOUND (tiling will use Haar Cascade)"
fi

if [ -f "$PROJECT_ROOT/server/models/haarcascade_frontalface_default.xml" ]; then
    SIZE=$(du -h "$PROJECT_ROOT/server/models/haarcascade_frontalface_default.xml" | cut -f1)
    echo "✅ haarcascade_frontalface_default.xml ($SIZE) - Fallback detector"
else
    echo "❌ haarcascade_frontalface_default.xml - MISSING"
fi

echo "======================================================================"
echo ""

# Final check
ARCFACE_OK=false
RETINAFACE_OK=false
HAAR_OK=false

if [ -f "$PROJECT_ROOT/server/models/arcfaceresnet100-8.onnx" ]; then
    ARCFACE_OK=true
fi

if [ -f "$PROJECT_ROOT/server/models/retinaface_r50_v1.onnx" ]; then
    RETINAFACE_OK=true
fi

if [ -f "$PROJECT_ROOT/server/models/haarcascade_frontalface_default.xml" ]; then
    HAAR_OK=true
fi

if [ "$ARCFACE_OK" = true ] && ([ "$RETINAFACE_OK" = true ] || [ "$HAAR_OK" = true ]); then
    echo "🎉 Essential models downloaded successfully!"
    if [ "$RETINAFACE_OK" = true ]; then
        echo "   ✅ Using RetinaFace for face detection (recommended)"
    elif [ "$HAAR_OK" = true ]; then
        echo "   ⚠️  Using Haar Cascade for face detection (fallback - less accurate)"
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Install dependencies: pip install -r server/requirements.txt"
    echo "  2. Run server: python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload"
    echo ""
else
    echo "⚠️  Some essential models failed to download."
    echo ""
    echo "Try downloading manually:"
    echo ""
    echo "1. ArcFace Model:"
    echo "   curl -L -o $PROJECT_ROOT/server/models/arcfaceresnet100-8.onnx \\"
    echo "     'https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx'"
    echo ""
    echo "2. RetinaFace Model (recommended):"
    echo "   curl -L -o $PROJECT_ROOT/server/models/retinaface_r50_v1.onnx \\"
    echo "     'https://huggingface.co/TheEeeeLin/HivisionIDPhotos_matting/resolve/main/retinaface-resnet50.onnx'"
    echo ""
    echo "3. Haar Cascade (fallback):"
    echo "   curl -o $PROJECT_ROOT/server/models/haarcascade_frontalface_default.xml \\"
    echo "     'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'"
    echo ""
fi
