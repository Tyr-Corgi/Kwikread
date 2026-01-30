#!/bin/bash
# Export fine-tuned TrOCR model to ONNX using Optimum
# Requires Python 3.12 or earlier (onnxruntime doesn't support Python 3.14 yet)
#
# PREREQUISITES:
# 1. Install Python 3.12: brew install python@3.12
# 2. Create venv: python3.12 -m venv venv_export
# 3. Install deps: pip install optimum[onnxruntime] transformers torch
#
# USAGE: ./export_with_optimum.sh

set -e

# Configuration
CHECKPOINT_DIR="./checkpoints/best"
OUTPUT_DIR="../Kwikread/Models"

echo "=== Exporting fine-tuned TrOCR to ONNX ==="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Output: $OUTPUT_DIR"

# Export using Optimum CLI
python -c "
from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import TrOCRProcessor

print('Loading model...')
model = ORTModelForVision2Seq.from_pretrained('$CHECKPOINT_DIR', export=True)
processor = TrOCRProcessor.from_pretrained('$CHECKPOINT_DIR')

print('Saving to ONNX...')
model.save_pretrained('$OUTPUT_DIR/onnx')
processor.save_pretrained('$OUTPUT_DIR/onnx')

print('Export complete!')
print('Files saved to: $OUTPUT_DIR/onnx/')
"

echo ""
echo "=== Next Steps ==="
echo "1. Copy encoder_model.onnx to $OUTPUT_DIR/encoder_model_quantized.onnx"
echo "2. Copy decoder_model_merged.onnx to $OUTPUT_DIR/decoder_model_merged_quantized.onnx"
echo "3. Run: cd ../Kwikread && dotnet build"
