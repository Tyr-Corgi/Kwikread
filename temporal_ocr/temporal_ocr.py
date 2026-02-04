#!/usr/bin/env python3
"""
Temporal OCR System for Handwritten Lists

Aggregates OCR results across multiple frames to produce stable, reliable
transcriptions even with motion blur, varying angles, and partial occlusions.

Usage:
    python temporal_ocr.py --input <video_or_folder> --out_dir out --engine paddle
    python temporal_ocr.py --input frames/ --out_dir results --max_frames 15 --use_consensus true
"""

import argparse
import sys
import warnings
from pathlib import Path

# Suppress specific warnings from dependencies (instead of blanket suppression)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

# Import from core modules
from core import (
    # Data classes (for external API compatibility)
    BoundingBox,
    DetectedLine,
    FrameResult,
    AggregatedLine,
    TemporalResult,
    # Preprocessing
    ImagePreprocessor,
    # Detection
    LineDetector,
    # Recognition
    OCREngine,
    EnsembleOCREngine,
    # Postprocessing
    FrameQualityAssessor,
    TemporalLineMatcher,
    TemporalAggregator,
    # Pipelines
    TemporalOCRPipeline,
    LineCropOCRPipeline,
)


def main():
    parser = argparse.ArgumentParser(
        description="Temporal OCR System for Handwritten Lists",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with temporal aggregation
  python temporal_ocr.py --input video.mp4 --out_dir out --engine paddle

  # Process folder of frames with consensus voting
  python temporal_ocr.py --input frames/ --out_dir out --max_frames 15 --use_consensus true

  # Single image mode (line-crop OCR)
  python temporal_ocr.py --input photo.jpg --out_dir out --mode single

  # Process folder of individual images
  python temporal_ocr.py --input images/ --out_dir out --mode single

  # Use ensemble voting (combines fine-tuned + base TrOCR models)
  python temporal_ocr.py --input photo.jpg --out_dir out --mode single --ensemble

  # Ensemble with custom base model
  python temporal_ocr.py --input photo.jpg --out_dir out --ensemble --ensemble_base trocr-base-handwritten
        """
    )

    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to video file, image, or folder of images/frames"
    )
    parser.add_argument(
        "--out_dir", "-o", default="out",
        help="Output directory (default: out)"
    )
    parser.add_argument(
        "--mode", "-m", choices=["temporal", "single"], default="auto",
        help="Processing mode: temporal (multi-frame), single (per-image), or auto-detect"
    )
    parser.add_argument(
        "--engine", "-e", choices=["paddle", "easyocr", "tesseract", "trocr", "ensemble"], default="paddle",
        help="OCR engine (default: paddle). Use 'trocr' for handwriting, 'ensemble' for multi-model voting."
    )
    parser.add_argument(
        "--model_path", default=None,
        help="Path to custom TrOCR model (only used with --engine trocr or ensemble)"
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Enable ensemble mode: combines fine-tuned and base TrOCR models with character-level voting"
    )
    parser.add_argument(
        "--ensemble_base", default="microsoft/trocr-large-handwritten",
        help="Base model for ensemble (default: microsoft/trocr-large-handwritten)"
    )
    parser.add_argument(
        "--ensemble_weights", default="0.6,0.4",
        help="Weights for ensemble voting as 'finetuned,base' (default: 0.6,0.4)"
    )
    parser.add_argument(
        "--lang", "-l", default="en",
        help="Language code (default: en)"
    )
    parser.add_argument(
        "--max_frames", type=int, default=15,
        help="Maximum frames to use for temporal aggregation (default: 15)"
    )
    parser.add_argument(
        "--min_frames", type=int, default=5,
        help="Minimum frames needed for temporal mode (default: 5)"
    )
    parser.add_argument(
        "--use_consensus", type=str, default="true",
        help="Use consensus voting (true/false, default: true)"
    )
    parser.add_argument(
        "--blur_threshold", type=float, default=100.0,
        help="Blur detection threshold (higher = stricter, default: 100)"
    )
    parser.add_argument(
        "--padding", type=int, default=12,
        help="Padding around line crops in pixels (default: 12)"
    )
    parser.add_argument(
        "--adaptive_thresh", action="store_true",
        help="Use adaptive thresholding for low contrast images"
    )
    parser.add_argument(
        "--no_deskew", action="store_true",
        help="Disable automatic deskewing"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--vision_verify", action="store_true",
        help="Enable vision LLM verification (requires Ollama with llava:7b)"
    )
    parser.add_argument(
        "--vision_model", default="llava:7b",
        help="Vision model for verification (default: llava:7b)"
    )
    parser.add_argument(
        "--vision_mode", choices=["verify_all", "verify_low", "primary"],
        default="verify_all",
        help="Vision verification mode (default: verify_all)"
    )

    args = parser.parse_args()

    # Determine mode
    input_path = Path(args.input)
    mode = args.mode

    if mode == "auto":
        if input_path.is_file():
            ext = input_path.suffix.lower()
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                mode = "temporal"
            else:
                mode = "single"
        elif input_path.is_dir():
            # Check if it looks like sequential frames
            files = list(input_path.iterdir())
            if len(files) > 5:
                mode = "temporal"
            else:
                mode = "single"
        else:
            print(f"Error: Input path does not exist: {args.input}")
            sys.exit(1)

    # Handle --ensemble flag: override engine to "ensemble"
    engine = args.engine
    if args.ensemble:
        engine = "ensemble"

    # Parse ensemble weights
    try:
        weights = tuple(float(w) for w in args.ensemble_weights.split(","))
        if len(weights) != 2:
            weights = (0.6, 0.4)
    except ValueError:
        weights = (0.6, 0.4)

    print(f"[Temporal OCR] Mode: {mode}")
    print(f"[Temporal OCR] Input: {args.input}")
    print(f"[Temporal OCR] Output: {args.out_dir}")
    print(f"[Temporal OCR] Engine: {engine}")
    if engine == "ensemble":
        print(f"[Temporal OCR] Ensemble base: {args.ensemble_base}")
        print(f"[Temporal OCR] Ensemble weights: {weights}")
    print()

    use_consensus = args.use_consensus.lower() in ["true", "yes", "1"]

    if mode == "temporal":
        # Temporal aggregation mode
        pipeline = TemporalOCRPipeline(
            engine=engine,
            max_frames=args.max_frames,
            min_frames=args.min_frames,
            use_consensus=use_consensus,
            blur_threshold=args.blur_threshold,
            padding=args.padding,
            lang=args.lang,
            adaptive_thresh=args.adaptive_thresh,
            model_path=args.model_path,
            ensemble_base_model=args.ensemble_base,
            ensemble_weights=weights
        )

        result = pipeline.process(args.input, args.out_dir)

        print("\n" + "="*60)
        print("AGGREGATED OCR RESULT")
        print("="*60)
        print(f"Frames processed: {result.frames_processed}")
        print(f"Frames used: {result.frames_used}")
        print(f"Lines detected: {len(result.lines)}")
        print(f"Method: {result.aggregation_method}")
        print("-"*60)
        print(result.full_text)
        print("="*60)

    else:
        # Single image mode
        pipeline = LineCropOCRPipeline(
            engine=engine,
            padding=args.padding,
            lang=args.lang,
            adaptive_thresh=args.adaptive_thresh,
            deskew=not args.no_deskew,
            model_path=args.model_path,
            vision_verify=args.vision_verify,
            vision_model=args.vision_model,
            vision_mode=args.vision_mode,
            ensemble_base_model=args.ensemble_base,
            ensemble_weights=weights
        )

        if input_path.is_file():
            result = pipeline.process_image(args.input, args.out_dir)

            print("\n" + "="*60)
            print("LINE-CROP OCR RESULT")
            print("="*60)
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Lines detected: {len(result['lines'])}")
                print(f"Skew angle: {result['skew_angle']:.1f}")
                print("-"*60)
                print(result['full_text'])
            print("="*60)

        else:
            results = pipeline.process_folder(args.input, args.out_dir)

            print("\n" + "="*60)
            print("BATCH PROCESSING COMPLETE")
            print("="*60)
            print(f"Images processed: {len(results)}")

            for r in results:
                if "error" not in r:
                    print(f"\n--- {Path(r['image_path']).name} ---")
                    print(r['full_text'])
            print("="*60)


if __name__ == "__main__":
    main()
