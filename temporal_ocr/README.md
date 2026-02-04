# Temporal OCR System

A high-accuracy handwritten grocery list OCR system optimized for real-world conditions. Uses TrOCR (Transformer-based OCR) for handwriting recognition with intelligent post-processing for grocery item correction.

## Performance

| Metric | Result |
|--------|--------|
| Fuzzy Match Accuracy | **97.3%** |
| Exact Match Accuracy | **83.8%** |
| Processing Speed | **3.79s / 16 images** |
| Test Coverage | **54 tests passing** |

Tested on real handwritten grocery lists with varied handwriting styles, two-column layouts, and challenging conditions.

## Features

- **Fine-Tuned TrOCR**: Custom model trained on 1,500+ IAM Handwriting Database samples plus user handwriting
- **Intelligent Correction**: 300+ grocery vocabulary with fuzzy matching and semantic validation
- **Two-Column Detection**: Automatic layout detection using DBSCAN clustering
- **Strikethrough Filtering**: Detects and excludes crossed-out items
- **Security**: Path validation with directory whitelisting to prevent traversal attacks
- **Socket API**: Fast persistent server keeps models loaded for instant inference

## Installation

```bash
cd temporal_ocr
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch (for TrOCR)
- ~4GB disk space for models (downloaded on first run)

## Quick Start

### Start the OCR Server

```bash
# Start server (keeps TrOCR model loaded)
python ocr_server.py serve &
```

### Process Images

```bash
# Process a single image
python ocr_server.py process grocery_list.jpg

# Process with output directory
python ocr_server.py process grocery_list.jpg --out_dir results
```

### Direct Usage (Without Server)

```bash
python temporal_ocr.py --input grocery_list.jpg --out_dir results --mode single --engine trocr
```

### Process Video (Temporal Mode)

```bash
python temporal_ocr.py --input video.mp4 --out_dir results --engine trocr
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input, -i` | (required) | Path to video, image, or folder |
| `--out_dir, -o` | `out` | Output directory |
| `--mode, -m` | `auto` | `temporal`, `single`, or `auto` |
| `--engine, -e` | `paddle` | OCR engine: `paddle`, `easyocr`, `tesseract`, `trocr` |
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

### OCR Pipeline

```
Input Image
    │
    ▼
┌─────────────────────────┐
│ EasyOCR Detection       │  Bounding box detection
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ DBSCAN Line Clustering  │  Group words by Y-position, detect columns
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ TrOCR Recognition       │  Handwriting-optimized transformer
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Strikethrough Filter    │  Remove crossed-out items
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Grocery Corrector       │  Fuzzy match to 300+ item vocabulary
└─────────────────────────┘
    │
    ▼
Output Text
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Detection | `ocr_server.py` | EasyOCR word detection, column splitting |
| Recognition | `ocr_server.py` | TrOCR inference with MPS/CUDA support |
| Strikethrough | `strikethrough_filter.py` | Morphological + Hough line detection |
| Correction | `grocery_corrector.py` | Conservative fuzzy matching |

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_accuracy.py

# Run performance benchmarks
pytest tests/test_performance.py
```

### Test Suite

| Test File | Coverage |
|-----------|----------|
| `test_accuracy.py` | OCR accuracy validation against ground truth |
| `test_corrector.py` | Grocery corrector fuzzy matching logic |
| `test_performance.py` | Speed benchmarks and regression tests |

## Configuration

Configuration files are located in `config/`:

```
config/
├── ocr_config.py          # Pipeline configuration dataclasses
└── vocabulary/
    ├── grocery.json       # Built-in vocabulary
    └── custom.json        # User additions
```

## Troubleshooting

### Poor Handwriting Recognition

- Ensure `--engine trocr` is set (not paddle/easyocr)
- Increase `--padding` to 16-20 for thick strokes
- Check image quality and lighting

### Columns Merging Together

The system automatically detects columns with gaps > 15% of image width. If columns are closer, they may merge.

### Wrong Reading Order

Check `annotated/` images to verify line detection. The system reads top-to-bottom within each column.

### Server Connection Issues

```bash
# Check if server is running
lsof -i :8765

# Restart server
pkill -f "ocr_server.py serve"
python ocr_server.py serve &
```

## Security

The OCR server includes path validation to prevent directory traversal attacks:

- **Directory Whitelist**: Only processes files within allowed directories
- **Extension Validation**: Only accepts image file extensions
- **Socket Input Validation**: Validates all paths received via socket API

## License

MIT License

## Model Training

The fine-tuned TrOCR model (`finetune/model_v3/final`) was trained on:

| Dataset | Samples | Description |
|---------|---------|-------------|
| IAM Handwriting Database | ~1,500 | Academic dataset with diverse handwriting styles |
| User samples (videotest2) | 21 | Custom grocery list handwriting |
| User samples (videotest3) | 16 | Additional grocery list handwriting |
| **Total** | **~1,537** | Mixed general + domain-specific |

The model automatically uses the fine-tuned weights when available, falling back to `microsoft/trocr-base-handwritten` otherwise.

### Training Scripts

```bash
# Prepare training data
python finetune/prepare_data.py

# Download IAM dataset (requires registration)
python finetune/download_iam.py

# Merge datasets
python finetune/merge_datasets.py

# Train model
python finetune/train_merged.py
```

## Acknowledgments

- Microsoft TrOCR for the base handwriting recognition model
- [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) for training data
- EasyOCR for text detection
- The grocery vocabulary is optimized for North American grocery items
