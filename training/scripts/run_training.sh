#!/bin/bash
# Kwikread TrOCR Training Script
# Run this to fine-tune the model on the preprocessed GNHK line-level data
#
# Usage: ./run_training.sh
#
# Prerequisites:
#   - Preprocessed dataset in ../../datasets/gnhk_lines/
#   - Virtual environment with dependencies installed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source ../.venv/bin/activate

echo "=============================================="
echo "Kwikread TrOCR Training"
echo "=============================================="
echo "Dataset: /Users/tygr/Desktop/repos/kwikread/datasets/gnhk_lines"
echo "Output: ./checkpoints_v2"
echo "=============================================="

# Run training
python train.py \
    --data_dir /Users/tygr/Desktop/repos/kwikread/datasets/gnhk_lines \
    --output_dir ./checkpoints_v2 \
    --epochs 30 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --warmup_steps 100 \
    --gradient_accumulation 2 \
    --eval_steps 100 \
    --save_steps 200 \
    --early_stopping 10 \
    --max_eval_samples 50 \
    --num_workers 2 \
    --no_amp \
    --export_onnx ./models/onnx_v2

echo ""
echo "=============================================="
echo "Training complete!"
echo "Checkpoints: ./checkpoints_v2"
echo "ONNX model: ./models/onnx_v2"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Copy ONNX models to Kwikread CLI:"
echo "   cp ./models/onnx_v2/* ../../Kwikread/Models/"
echo ""
echo "2. Build and test Kwikread:"
echo "   cd ../../Kwikread && dotnet build && dotnet run -- process test_image.png"
