# Temporal OCR System

A handwritten grocery list OCR system optimized for real-world conditions. Uses TrOCR for handwriting recognition with optional Vision LLM verification to correct recognition errors.

## Current Status

**Tested Accuracy: 81% (17/21 items correct)** on handwritten grocery lists.

### What Works Well
- Handwriting recognition via TrOCR (microsoft/trocr-large-handwritten)
- Two-column layout detection and separation
- Vision LLM verification corrects common TrOCR mistakes:
  - "out milk" → "oat milk"
  - "problem Bars" → "PROTEIN BARS"
  - "Apple since" → "apple sauce"
  - "Trail unit" → "Trail Mix"
  - Truncated text like "Cheese" → "cheese sticks"

### Known Limitations
- Some words still misread ("french fries" → "French tries")
- Crossed-out/scribbled text not yet filtered
- Requires Ollama with LLaVA for vision verification

## Installation

```bash
cd temporal_ocr
pip install -r requirements.txt
```

### For Vision Verification (Recommended)
```bash
# Install Ollama
brew install ollama

# Pull LLaVA model (4.7GB)
ollama pull llava:7b
```

## Quick Start

### Basic Usage (TrOCR only)
```bash
python temporal_ocr.py --input grocery_list.jpg --out_dir results --mode single --engine trocr
```

### With Vision Verification (Recommended)
```bash
python temporal_ocr.py --input grocery_list.jpg --out_dir results --mode single --engine trocr --vision_verify
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
| `--vision_verify` | off | Enable vision LLM verification |
| `--vision_model` | `llava:7b` | Ollama vision model to use |
| `--vision_mode` | `verify_all` | `verify_all`, `verify_low`, or `primary` |
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
  "full_text": "Rice\noat milk\nPROTEIN BARS\n...",
  "skew_angle": 0.7,
  "lines": [
    {
      "index": 0,
      "bbox": [113, 509, 146, 66],
      "text": "Rice",
      "confidence": 0.99,
      "crop_path": "results/crops/grocery_list/line_00.png"
    }
  ]
}
```

## Architecture

### OCR Pipeline
1. **Text Detection**: EasyOCR detects word bounding boxes
2. **Line Grouping**: DBSCAN clusters words by Y-position, splits columns by X-gaps
3. **Line Cropping**: Each line cropped with padding for OCR
4. **TrOCR Recognition**: Handwriting-optimized transformer model
5. **Vision Verification** (optional): LLaVA corrects semantic errors

### Vision Verification Strategy
The vision verification module uses LLaVA to verify TrOCR output:
- If vision confidence > 0.9 and texts differ: trust vision (semantic understanding)
- If texts are 95%+ similar: both agree, use TrOCR
- Otherwise: use higher confidence result

This catches errors where TrOCR produces phonetically similar but semantically wrong text (e.g., "out milk" vs "oat milk").

## Supported OCR Engines

| Engine | Best For | Notes |
|--------|----------|-------|
| `trocr` | Handwriting | Recommended for grocery lists |
| `paddle` | Printed text | Good general-purpose |
| `easyocr` | Mixed content | Slower but robust |
| `tesseract` | Typed text only | Not recommended for handwriting |

## Testing Vision OCR

To test vision model accuracy on crop images:
```bash
python test_vision_ocr.py llava:7b
```

## Troubleshooting

### "Vision verification disabled"
Ensure Ollama is running with LLaVA:
```bash
ollama serve  # In one terminal
ollama pull llava:7b  # If not already pulled
```

### Poor handwriting recognition
- Use `--engine trocr` (not paddle/easyocr)
- Enable `--vision_verify` for correction
- Increase `--padding` to 16-20 for thick strokes

### Columns merging together
The system automatically detects columns with gaps > 100px. If columns are closer, they may merge.

### Wrong reading order
Check `annotated/` images to verify line detection. The system reads top-to-bottom within each column.

## Requirements

- Python 3.8+
- PyTorch (for TrOCR)
- Ollama + LLaVA (for vision verification)
- See `requirements.txt` for full list
