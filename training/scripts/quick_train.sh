#!/bin/bash
# Quick Training Script for TrOCR Fine-tuning
# ============================================
# This script sets up the environment and starts training.
#
# Usage:
#   ./quick_train.sh                    # Train with default settings
#   ./quick_train.sh --epochs 5         # Custom epochs
#   ./quick_train.sh --data_dir /path   # Custom data directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATASETS_DIR="$PROJECT_DIR/datasets"

echo "=================================================="
echo "TrOCR Fine-tuning for Kwikread"
echo "=================================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required but not found"
    exit 1
fi

echo "Python: $(python3 --version)"

# Create virtual environment if it doesn't exist
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r "$PROJECT_DIR/requirements.txt"

# Check if GNHK dataset exists
GNHK_DIR="$DATASETS_DIR/gnhk"
if [ ! -d "$GNHK_DIR" ] || [ -z "$(ls -A $GNHK_DIR 2>/dev/null)" ]; then
    echo ""
    echo "=================================================="
    echo "GNHK dataset not found!"
    echo "=================================================="
    echo ""
    echo "Please download the GNHK dataset:"
    echo "  python scripts/download_gnhk.py --output $GNHK_DIR"
    echo ""
    echo "Or manually download from:"
    echo "  https://drive.google.com/drive/folders/1KXu55SBzyyZf0Ek-F6Kudre1tWd9dDC2"
    echo ""
    echo "After downloading, run this script again."
    exit 1
fi

# Start training
echo ""
echo "=================================================="
echo "Starting training..."
echo "=================================================="
echo ""

cd "$SCRIPT_DIR"

python train.py \
    --data_dir "$GNHK_DIR" \
    --output_dir "$PROJECT_DIR/checkpoints" \
    --log_dir "$PROJECT_DIR/logs" \
    "$@"

echo ""
echo "=================================================="
echo "Training complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Export to ONNX:"
echo "     python scripts/export_onnx.py --checkpoint ./checkpoints/best --output ./models"
echo ""
echo "  2. Integrate with kwikread:"
echo "     python scripts/export_onnx.py --checkpoint ./checkpoints/best --integrate ../Kwikread/Models"
echo ""
echo "  3. Rebuild kwikread:"
echo "     cd ../Kwikread && dotnet build"
