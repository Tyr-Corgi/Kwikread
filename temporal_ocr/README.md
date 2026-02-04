# Temporal OCR System

A high-accuracy handwritten text OCR system optimized for real-world conditions. Uses a fine-tuned TrOCR (Transformer-based OCR) model with batch inference for fast, accurate handwriting recognition.

## Performance

| Metric | Result |
|--------|--------|
| **Accuracy** | **100%** (37/37 test items) |
| **Processing Speed** | **2.6s** per document (~20 lines) |
| **Batch Throughput** | **~140ms** per line crop |

Tested on real handwritten grocery lists with varied handwriting styles, two-column layouts, and challenging conditions.

## Features

- **Fine-Tuned TrOCR**: Custom model trained on 1,500+ IAM Handwriting samples plus domain-specific data
- **Batch Inference**: Process all lines in a single forward pass for maximum throughput
- **FP16 Acceleration**: Half-precision inference on Apple Silicon (MPS) or CUDA GPUs
- **Two-Column Detection**: Automatic layout detection using DBSCAN clustering
- **Strikethrough Filtering**: Detects and excludes crossed-out items
- **Grocery Correction**: 300+ item vocabulary with fuzzy matching
- **Socket API**: Persistent server keeps models loaded for instant inference

## Installation

```bash
cd temporal_ocr
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+ (for TrOCR and MPS support)
- ~4GB disk space for models (downloaded on first run)

## Quick Start

### Option 1: Direct Processing

```bash
# Process a single image
python temporal_ocr.py --input grocery_list.jpg --out_dir results --engine trocr

# Process a folder of images
python temporal_ocr.py --input ./images/ --out_dir results --mode single --engine trocr
```

### Option 2: Server Mode (Recommended for Multiple Requests)

```bash
# Start server (keeps model loaded in memory)
python ocr_server.py serve &

# Process images via server (instant inference)
python ocr_server.py process grocery_list.jpg --out_dir results
```

### Process Video (Temporal Aggregation)

```bash
python temporal_ocr.py --input video.mp4 --out_dir results --engine trocr
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input, -i` | (required) | Path to video, image, or folder |
| `--out_dir, -o` | `out` | Output directory |
| `--mode, -m` | `auto` | `temporal`, `single`, or `auto` |
| `--engine, -e` | `trocr` | OCR engine: `trocr`, `paddle`, `easyocr` |
| `--padding` | `12` | Padding around line crops (pixels) |
| `--no_deskew` | `false` | Disable automatic deskewing |

## Output Format

```
results/
├── annotated/              # Images with detected line boxes
│   └── grocery_list.png
├── crops/                  # Individual line crops
│   └── grocery_list/
│       ├── line_00.png
│       ├── line_01.png
│       └── ...
└── grocery_list_result.json
```

### JSON Output

```json
{
  "image_path": "grocery_list.jpg",
  "full_text": "milk\noat milk\nprotein bars\n...",
  "skew_angle": 0.7,
  "lines": [
    {
      "index": 0,
      "bbox": [113, 509, 146, 66],
      "text": "milk",
      "confidence": 0.99,
      "crop_path": "results/crops/grocery_list/line_00.png"
    }
  ]
}
```

## Architecture

```
Input Image
    │
    ▼
┌─────────────────────────┐
│ EasyOCR Detection       │  Word-level bounding boxes
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Line Clustering         │  Group words into lines, detect columns
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ TrOCR Batch Inference   │  FP16, greedy decode, all lines at once
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Post-Processing         │  Strikethrough filter, grocery correction
└─────────────────────────┘
    │
    ▼
Output JSON
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Core Pipeline | `core/` | Modular OCR pipeline components |
| Recognition | `core/recognition.py` | TrOCR with batch inference |
| Detection | `core/detection.py` | Line detection and clustering |
| Server | `ocr_server.py` | Socket API with warm model |
| CLI | `temporal_ocr.py` | Command-line interface |

## Running Tests

```bash
# Run accuracy benchmark
python scripts/test_accuracy.py

# Run full test suite
pytest tests/ -v

# Run specific test
pytest tests/test_accuracy.py
```

## Project Structure

```
temporal_ocr/
├── core/                   # Core OCR modules
│   ├── recognition.py      # TrOCR engine with batch inference
│   ├── detection.py        # Line detection
│   ├── preprocessing.py    # Image preprocessing
│   ├── postprocessing.py   # Result aggregation
│   ├── video.py            # Pipeline orchestration
│   └── utils.py            # Data structures
├── config/                 # Configuration files
├── finetune/               # Fine-tuned model weights
│   └── model_v4/final/     # Current production model
├── scripts/                # Utility scripts
│   ├── test_accuracy.py    # Accuracy benchmark
│   └── benchmark_batch.py  # Performance benchmark
├── tests/                  # Unit tests
├── ocr_server.py           # Socket server for warm inference
├── temporal_ocr.py         # CLI entry point
├── grocery_corrector.py    # Fuzzy matching for grocery items
├── strikethrough_filter.py # Crossed-out text detection
└── requirements.txt        # Dependencies
```

## Performance Optimization

The system achieves <3 second processing through:

1. **Batch Inference**: All line crops processed in one forward pass
2. **FP16 Precision**: 2x memory reduction, faster on MPS/CUDA
3. **Greedy Decoding**: `num_beams=1` for maximum speed
4. **Minimal Preprocessing**: Let TrOCRProcessor handle resizing

### Benchmarks

| Configuration | Time (21 lines) | Accuracy |
|--------------|-----------------|----------|
| Sequential + Beam Search | ~11s | 100% |
| Batch + Beam Search | ~4s | 100% |
| Batch + Greedy + FP16 | **2.6s** | **100%** |

## Model Training

The fine-tuned model (`finetune/model_v4/final`) was trained on:

| Dataset | Samples |
|---------|---------|
| IAM Handwriting Database | ~1,500 |
| Domain-specific samples | ~40 |
| **Total** | **~1,540** |

Training scripts are available in `finetune/`.

## Troubleshooting

### Slow First Inference

The first inference is slower due to model loading. Use server mode (`ocr_server.py serve`) to keep the model warm.

### Out of Memory

Reduce batch size or ensure FP16 is enabled (automatic on MPS/CUDA).

### Poor Recognition

- Ensure `--engine trocr` is set
- Check image quality and lighting
- Increase `--padding` for thick strokes

## License

MIT License
