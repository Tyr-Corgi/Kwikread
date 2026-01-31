# Temporal OCR System

A robust OCR system designed for real-world conditions: motion blur, varying angles, partial occlusions, and uneven lighting. Instead of relying on single-frame OCR, this system aggregates results across multiple frames for stable, reliable transcriptions.

## Features

- **Temporal Aggregation**: Combines OCR results from 5-15 frames for stable output
- **Per-Line Cropping**: Detects and crops individual text lines before OCR
- **Quality-Based Frame Selection**: Automatically rejects blurry/dark frames
- **Consensus Voting**: Character-level voting with edit-distance alignment
- **Automatic Deskewing**: Corrects tilted text before recognition
- **Multiple OCR Engines**: PaddleOCR (preferred), EasyOCR, or Tesseract

## Installation

```bash
cd temporal_ocr
pip install -r requirements.txt
```

### OCR Engine Notes

- **PaddleOCR** (recommended): Best for handwriting, includes text detection
  ```bash
  pip install paddlepaddle paddleocr
  ```
  Models download automatically on first use (~100MB)

- **EasyOCR** (alternative): Good fallback, slightly slower
  ```bash
  pip install easyocr
  ```

- **Tesseract** (fallback only): Weak for handwriting
  ```bash
  brew install tesseract  # macOS
  pip install pytesseract
  ```

## Usage

### Temporal Mode (Video or Frame Folder)

Process a video file:
```bash
python temporal_ocr.py --input video.mp4 --out_dir results --engine paddle
```

Process a folder of sequential frames:
```bash
python temporal_ocr.py --input frames/ --out_dir results --max_frames 20 --use_consensus true
```

### Single Image Mode

Process a single image:
```bash
python temporal_ocr.py --input photo.jpg --out_dir results --mode single
```

Process a folder of individual images:
```bash
python temporal_ocr.py --input images/ --out_dir results --mode single
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input, -i` | (required) | Path to video, image, or folder |
| `--out_dir, -o` | `out` | Output directory |
| `--mode, -m` | `auto` | `temporal`, `single`, or `auto` |
| `--engine, -e` | `paddle` | OCR engine: `paddle`, `easyocr`, `tesseract` |
| `--lang, -l` | `en` | Language code |
| `--max_frames` | `15` | Max frames for temporal aggregation |
| `--min_frames` | `5` | Min frames needed |
| `--use_consensus` | `true` | Enable consensus voting |
| `--blur_threshold` | `100.0` | Blur detection threshold (higher = stricter) |
| `--padding` | `12` | Padding around line crops (pixels) |
| `--adaptive_thresh` | `false` | Use adaptive thresholding |
| `--no_deskew` | `false` | Disable automatic deskewing |

## Outputs

```
out/
├── annotated/           # Images with detected line boxes drawn
│   ├── frame_0000.png
│   └── frame_0001.png
├── crops/               # Individual line crops per frame
│   ├── frame_0000/
│   │   ├── line_00.png
│   │   └── line_01.png
│   └── frame_0001/
├── frame_results.json   # Per-frame detection results
└── aggregated_results.json  # Final temporal aggregation
```

### JSON Output Format

**frame_results.json**:
```json
[
  {
    "frame_index": 0,
    "quality_score": 245.7,
    "is_valid": true,
    "skew_angle": -2.1,
    "lines": [
      {
        "index": 0,
        "bbox": [50, 100, 400, 35],
        "text": "Milk",
        "confidence": 0.95
      }
    ]
  }
]
```

**aggregated_results.json**:
```json
{
  "full_text": "Milk\nBread\nEggs",
  "frames_processed": 30,
  "frames_used": 12,
  "aggregation_method": "consensus",
  "lines": [
    {
      "line_index": 0,
      "final_text": "Milk",
      "confidence": 0.92,
      "contributing_frames": [0, 2, 5, 7, 9]
    }
  ]
}
```

## Tuning Guide

### For Blurry Frames
- **Lower `--blur_threshold`** to accept more frames (default: 100)
- Try values 50-80 if too many frames are rejected

### For Low Contrast / Handwriting
- Add `--adaptive_thresh` for very faint text
- Increase `--padding` to 16-20 for thicker strokes

### For Tilted Text
- Deskewing is automatic; disable with `--no_deskew` if causing issues

### For Non-English Text
- Use `--lang` with appropriate code: `ch` (Chinese), `japan`, `korean`, etc.
- PaddleOCR supports 80+ languages

### For Very Short Lists (<5 lines)
- Use `--mode single` instead of temporal
- Or lower `--min_frames` to 3

## How It Works

### 1. Frame Quality Assessment
Each frame is scored on:
- **Blur metric**: Variance of Laplacian (sharp edges = higher score)
- **Brightness**: Rejects too dark (<30) or too bright (>225)
- **Text presence**: Must detect at least 1 text region

### 2. Preprocessing Pipeline
- Grayscale conversion
- Illumination normalization (removes shadows)
- CLAHE contrast enhancement
- Bilateral filter denoising (preserves edges)
- Optional adaptive thresholding

### 3. Line Detection
- Uses PaddleOCR's DBNet for text region detection
- Groups word boxes into lines using DBSCAN on y-centers
- Handles multi-column and tilted text

### 4. Cross-Frame Line Matching
- Matches lines by normalized y-position (15% tolerance)
- Uses text similarity as secondary signal
- Hungarian algorithm for optimal assignment

### 5. Temporal Aggregation
Two strategies:
1. **Best-of-N**: Select highest-confidence recognition per line
2. **Consensus Voting**: Character-level weighted voting
   - Aligns candidate strings using edit distance
   - Confidence-weighted character voting
   - Falls back to best-of-N if consensus is weak

## Performance Tips

- For fastest processing, use `--engine paddle` with GPU:
  ```bash
  pip install paddlepaddle-gpu
  ```

- For memory-constrained systems, use `--engine easyocr`

- Limit frames with `--max_frames 10` for faster processing

## Troubleshooting

### "No OCR engine available"
Install at least one OCR engine:
```bash
pip install paddlepaddle paddleocr  # Recommended
# OR
pip install easyocr
```

### "Too many frames rejected"
Lower the blur threshold:
```bash
python temporal_ocr.py --input video.mp4 --blur_threshold 50
```

### Poor handwriting recognition
Try adaptive thresholding:
```bash
python temporal_ocr.py --input video.mp4 --adaptive_thresh
```

### Wrong reading order
Check the annotated output images to verify line detection. Adjust `--padding` if lines are merging.
