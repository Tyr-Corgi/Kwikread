# Kwikread - Self-Hosted Multi-Line Handwriting OCR

**Status: ✅ FUNCTIONAL - Base model working with C# ONNX inference**

---

## Mission Statement

**Build in-house handwriting OCR capability that rivals cloud services.**

This is NOT a cloud API wrapper. This is a fully local, self-hosted solution that gives you:
- ✅ Complete data privacy (nothing leaves your machine)
- ✅ Zero ongoing costs (no API fees)
- ✅ Full control over the model and pipeline
- ✅ Self-learning system that improves with your corrections

## What This Project Does

Kwikread reads handwritten lists from pictures and outputs structured text/CSV data. It:

1. **Detects** whether an image has single or multiple lines
2. **Segments** multi-line images into individual text lines
3. **Recognizes** handwriting using TrOCR (Transformer-based OCR)
4. **Learns** from your corrections to build custom training data
5. **Exports** results to CSV or structured formats

---

## Current Project Status (January 2026)

### ✅ COMPLETE: C# ONNX Inference Working

**Model**: `microsoft/trocr-base-handwritten` (pre-trained, 1.3GB)
**Status**: C# inference now matches Python output

| Test Image | Expected | C# Output | Status |
|------------|----------|-----------|--------|
| "DFI Rocks" | "DFI Rocks" | "DFL Rocks ." | ✅ Good |
| "what I hope to achieve during this" | "what I hope to achieve during this" | "what I hope ! to achieve during this" | ✅ Good |
| "lights and continue" | "lights and continue" | "lights and , consume ." | ⚠️ Partial |

**Assessment**: The base model produces reasonable OCR output. Minor errors occur but text is readable and usable. Fine-tuning on domain-specific data would improve accuracy.

### Small Model Fine-tuning (Experimental)

Attempted fine-tuning on GNHK dataset with the small model:
- **Training Epochs**: 13/15 (early stopping)
- **Final Loss**: 2.64 (down from 7.4)
- **CER**: ~77% (too high for production use)
- **Conclusion**: Small model has insufficient capacity; use base model instead

### ✅ What's Complete & Working

| Component | Status | Notes |
|-----------|--------|-------|
| **C# Inference Pipeline** | ✅ Working | ONNX Runtime with merged decoder + KV-cache |
| **TrOCR Base Model** | ✅ Working | Output matches Python inference |
| **Multi-line Detection** | ✅ Working | Automatic full-page processing |
| **Line Segmentation** | ✅ Working | 2 methods (projection + chunking) |
| **Self-learning System** | ✅ Working | Thread-safe CorrectionTracker |
| **CSV Export** | ✅ Working | Batch processing supported |
| **Python Training Pipeline** | ✅ Working | train.py, dataset.py, export scripts |
| **ONNX Export** | ✅ Working | Supports both merged and separate decoder formats |

### ⚠️ Areas for Improvement

| Issue | Description | Next Steps |
|-------|-------------|------------|
| **Multi-line alignment** | Chunk boundaries may not match text lines on complex pages | Stronger peak detection or ML-based line detector |
| **Accuracy on Cursive** | Base model struggles with cursive | Fine-tune on cursive samples |
| **Domain Adaptation** | GNHK dataset may not match your use case | Collect domain-specific data |
| **Processing Speed** | ~4-7 seconds per line | Consider GPU acceleration |

### Full-Page Line Segmentation (LineSegmenter.cs)

Improvements added so full-page images segment better:

- **Smoothed projection** – Horizontal projection is smoothed (5-row moving average) to reduce single-row noise and false line boundaries.
- **Peak-based subdivision** – When the threshold method finds 0 regions or one huge region (>60% of page), the segmenter subdivides by **projection peaks** (local maxima = gaps between lines) instead of going straight to fixed-height chunking.
- **Adaptive fallback** – When projection still fails, fixed-height chunking uses **adaptive chunk height** (estimated from image height with ~70px per line) so chunk count matches typical line count better.

For best full-page results, use clear handwriting on a clean background; complex layouts or low contrast may still fall back to chunking. Future options: Sauvola/local binarization, contour-based line detection, or a small ML line detector.

---

## Recent Fixes (January 2026)

### Critical C# Inference Bugs Fixed

1. **Model Config Mismatch** (`config.json`)
   - Issue: Config described small model (6 layers, 256 d_model) but ONNX was base model (12 layers, 1024 d_model)
   - Fix: Updated config.json with correct base model dimensions
   - Impact: KV-cache tensors now have correct shapes

2. **Vocabulary Mismatch** (`vocab.json`)
   - Issue: Vocab was small model's 64k SentencePiece tokens (▁ markers)
   - Fix: Extracted correct 50,265 token RoBERTa vocab from base model (Ġ markers)
   - Impact: Token decoding now produces correct text

3. **Image Preprocessing Mismatch** (`OcrEngine.cs`)
   - Issue: C# preserved aspect ratio with white padding; Python squishes to 384x384
   - Fix: Changed to squish-resize matching HuggingFace ViTImageProcessor behavior
   - Impact: Wide images now processed correctly (C# output matches Python)

### Earlier Fixes

4. **Image Mutation Bug** (`OcrEngine.cs`)
   - Issue: `image.Mutate()` was destroying original images during preprocessing
   - Fix: Changed to `image.Clone(x => x.Resize(...))` to preserve originals

5. **File Locking Race Condition** (`CorrectionTracker.cs`)
   - Issue: Concurrent writes could corrupt corrections.json
   - Fix: Added `lock` object for thread safety and `FileShare.None` for exclusive file access

6. **KV-Cache Implementation** (`OcrEngine.cs`)
   - Issue: Merged decoder model requires KV-cache inputs
   - Fix: Implemented proper cache initialization and management:
     - `use_cache_branch=false` for first token (compute cache)
     - `use_cache_branch=true` for subsequent tokens (use cache)
     - Encoder cache stored once, decoder cache grows each step

4. **Iterator Consumption Bug**
   - Issue: ONNX results iterator was being consumed twice, causing empty cache
   - Fix: Convert to list immediately with `results.ToList()` before any iteration

5. **Tensor Data Corruption**
   - Issue: `DenseTensor` was corrupting cached arrays
   - Fix: Clone arrays before passing to tensor constructor

---

## Technical Details

### TrOCR Model Architecture

**Current**: Pre-trained `microsoft/trocr-base-handwritten`:

| Component | Details |
|-----------|---------|
| **Encoder** | ViT (Vision Transformer), 12 layers, 768 hidden size |
| **Decoder** | TrOCR decoder, 12 layers, 16 attention heads, 1024 d_model |
| **Image Size** | 384x384 pixels (squish resize) |
| **Vocabulary** | 50,265 tokens (RoBERTa BPE tokenizer) |
| **Model Size** | ~1.3GB total (encoder: 344MB, decoder: 990MB) |
| **ONNX Format** | Merged decoder with `use_cache_branch` |

**Alternative**: Fine-tuned `trocr-small-handwritten` (backup_20260128_174155/):

| Component | Details |
|-----------|---------|
| **Encoder** | DeiT (Data-efficient ViT), 12 layers, 384 hidden size |
| **Decoder** | TrOCR decoder, 6 layers, 8 attention heads, 256 d_model |
| **Model Size** | ~250MB total (encoder: 87MB, decoder: 159MB) |
| **Note** | Limited accuracy - use base model for better results |

### Performance Benchmarks

| Metric | Current Performance |
|--------|-------------------|
| **Single-line processing** | 4-7 seconds |
| **Multi-line per line** | ~8 seconds |
| **Line detection** | < 1 second |
| **Model loading** | 5-10 seconds (once at startup) |
| **Memory usage** | ~2GB RAM |
| **Accuracy** | Limited without fine-tuning (hallucinates on complex images) |

---

## Quick Start

### Installation

```bash
# Clone the repository
cd /Users/tygr/Desktop/Repos/kwikread

# Build the C# project
cd Kwikread && dotnet build

# Models should be in Kwikread/Models/
# Required files:
#   - encoder_model.onnx (329MB)
#   - decoder_model.onnx (944MB)
#   - vocab.json (780KB)
```

### Basic Usage

```bash
# Process a single line of handwriting
./Kwikread/bin/Debug/net8.0/Kwikread process image.png

# Process a full page with multiple lines
./Kwikread/bin/Debug/net8.0/Kwikread process grocery_list.jpg

# Interactive mode (review and correct)
./Kwikread/bin/Debug/net8.0/Kwikread process image.png -r

# Export to CSV
./Kwikread/bin/Debug/net8.0/Kwikread process image.png -o output.csv

# Check learning statistics
./Kwikread/bin/Debug/net8.0/Kwikread stats
```

---

## Project Structure

```
kwikread/
├── Kwikread/                 # C# .NET application
│   ├── Program.cs            # CLI interface
│   ├── OcrEngine.cs          # TrOCR inference engine (with KV-cache)
│   ├── LineSegmenter.cs      # Multi-line detection & segmentation
│   ├── CorrectionTracker.cs  # Thread-safe self-learning system
│   └── Models/               # ONNX models (fine-tuned small model)
│       ├── encoder_model.onnx         # 87MB - DeiT encoder
│       ├── decoder_model.onnx         # 159MB - TrOCR decoder (first token)
│       ├── decoder_with_past_model.onnx # 154MB - Decoder with KV cache
│       ├── vocab.json                 # 64,002 token vocabulary
│       ├── config.json                # Model configuration
│       └── backup_base_model/         # Original Xenova base model
├── training/                 # Python training pipeline
│   ├── scripts/
│   │   ├── train.py          # Fine-tuning script
│   │   ├── export_onnx.py    # Model export
│   │   └── dataset.py        # Data loading
│   └── checkpoints_gnhk_lines/   # Training checkpoints
├── datasets/                 # Sample handwriting data
│   ├── gnhk_lines/           # Pre-segmented lines with labels
│   │   ├── train/            # 5,586 training samples
│   │   └── test/             # 1,806 test samples
│   └── gnhk/                 # Full-page images with manifests
└── README.md                 # This file
```

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                  Input Image                         │
│         (single line or full page)                   │
└─────────────────────┬────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────┐
        │   Multi-line Detection  │
        │   • Height analysis     │
        │   • Automatic routing   │
        └──────────┬──────────────┘
                   │
        ┌──────────▼──────────────────┐
        │   Line Segmentation         │
        │   • Projection analysis     │
        │   • Fixed chunking fallback │
        └──────────┬──────────────────┘
                   │
        ┌──────────▼──────────────────┐
        │    TrOCR Processing         │
        │    • ViT encoder            │
        │    • Decoder with KV-cache  │
        │    • Autoregressive gen     │
        └──────────┬──────────────────┘
                   │
        ┌──────────▼──────────────────┐
        │   Interactive Review        │
        │   • Accept or correct       │
        │   • Build training data     │
        └──────────┬──────────────────┘
                   │
        ┌──────────▼──────────────────┐
        │   Output: Text/CSV          │
        │   Training data stored      │
        └─────────────────────────────┘
```

---

## Development Roadmap

### Phase 1: Infrastructure ✅ COMPLETE
- [x] Multi-line detection and segmentation
- [x] TrOCR integration with ONNX Runtime
- [x] KV-cache implementation for merged decoder
- [x] CLI with interactive review mode
- [x] Thread-safe self-learning correction tracker
- [x] CSV export and batch processing
- [x] Python training pipeline

### Phase 2: Accuracy Improvement ⚠️ NEEDS MORE WORK
- [x] Set up Python training environment
- [x] Create line-level training data with labels
- [x] Attempt fine-tuning (model collapsed - needs hyperparameter tuning)
- [x] Deploy pre-trained Xenova model (working but imperfect)
- [x] Switch to small model for faster training on MPS
- [x] Fix YAML config type conversion bug
- [x] Fine-tune small model on GNHK (13 epochs, CER: 77%)
- [x] Export fine-tuned model to ONNX
- [x] Update C# engine for Optimum ONNX format (separate decoders)
- [ ] **Try base model** - small model has limited capacity
- [ ] **Extend training** - 30+ epochs may help
- [ ] **Collect domain-specific data** - GNHK may not match your use case
- [ ] Target: 80%+ accuracy on test set (CER < 20%)

### Phase 3: Advanced Features 📋 PLANNED
- [ ] GPU acceleration support
- [ ] Real-time processing mode
- [ ] Multiple handwriting style models
- [ ] REST API for integration
- [ ] Web UI for batch processing

---

## Technology Stack

- **Language**: C# (.NET 8.0) for inference, Python for training
- **Model**: TrOCR (microsoft/trocr-base-handwritten via Xenova)
- **Runtime**: ONNX Runtime (cross-platform, optimized)
- **ML Framework**: Transformers, Optimum (HuggingFace)
- **Image Processing**: SixLabors.ImageSharp
- **Dataset**: GNHK handwriting dataset (7,392 line samples)

---

## Improving Accuracy

### Current Accuracy Assessment

The pre-trained model works well on **clean, white-background, print-style** handwriting but struggles with:
- Grid/lined paper backgrounds
- Cursive handwriting styles
- Low contrast or noisy images

| Image Type | Example | Accuracy |
|------------|---------|----------|
| Clean print on white | "DFI Rocks" | Fair ("DFI races .") |
| Cursive on grid paper | "lights and continue" | Poor (hallucination) |

**Fine-tuning on GNHK dataset is required for production use.**

### Option A: Fine-tune with Config Files (Recommended)

Pre-configured YAML files are available in `training/configs/`:

```bash
cd training/scripts

# Conservative config (recommended first attempt)
python train.py --config ../configs/train_conservative.yaml

# Ultra-safe config (if conservative still collapses)
python train.py --config ../configs/train_safe.yaml
```

**Conservative config settings** (`train_conservative.yaml`):
- Learning rate: 1e-5 (5x lower than default)
- Warmup: 1000 steps
- Gradient accumulation: 4 (effective batch = 16)
- Early stopping: 10 evaluations

**Ultra-safe config settings** (`train_safe.yaml`):
- Learning rate: 5e-6 (10x lower than default)
- Warmup: 2000 steps
- Gradient accumulation: 8 (effective batch = 16)
- AMP disabled for stability

### Option B: Manual Hyperparameter Tuning

```bash
cd training/scripts
python train.py \
  --data_dir ../../datasets/gnhk_lines \
  --epochs 20 \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --warmup_steps 1000 \
  --gradient_accumulation 4 \
  --early_stopping 10
```

Key settings to prevent model collapse:
- **Lower learning rate**: 1e-5 or 5e-6 (NOT 5e-5)
- **Extended warmup**: 1000+ steps
- **Gradient accumulation**: 4-8 for effective larger batches
- **Gradient clipping**: 0.5 or lower

### Option B: Collect Domain-Specific Data

Use the self-learning system:
1. Process images with `--review` flag
2. Correct mistakes when prompted
3. Corrections are saved to `~/.config/kwikread/corrections.json`
4. Export training data with `kwikread export`
5. Re-train model with your corrections

### Option C: Try the Base Model

The fine-tuned small model has limited capacity. To use the base model:

```bash
# Restore base model (from backup)
cd Kwikread/Models
cp backup_base_model/* .

# Note: Base model uses merged decoder format (single decoder_model.onnx)
# The C# engine auto-detects this and uses the appropriate code path
```

### Option D: Alternative Models

Consider these if TrOCR doesn't meet accuracy needs:
- **trocr-large-handwritten** - Larger model, potentially better accuracy
- **PaddleOCR** - Strong handwriting support
- **EasyOCR** - Simple API, pre-trained models

---

## Why In-House?

**Data Privacy**: Medical forms, financial documents, personal lists - keep it local
**Cost Control**: No per-image fees, no usage limits, no surprises
**Customization**: Fine-tune for YOUR handwriting, YOUR domain
**Offline Operation**: Works without internet, on-premise servers, air-gapped systems
**Full Control**: Own the entire stack, modify as needed

---

## Benchmark Reference

**Cloud services like Azure AI Vision achieve 93-100% accuracy.** That's our target for in-house capability.

**Current Status**: The infrastructure is complete and functional. The pre-trained model works well on clean images but produces hallucinations on complex handwriting (grid paper, cursive). **Fine-tuning is essential** - use the config files in `training/configs/` to train on the GNHK dataset (7,392 labeled samples).

---

## License

[Add your license here]
