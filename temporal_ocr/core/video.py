"""
Video and temporal processing functions for Temporal OCR.

Contains the main pipeline classes for multi-frame temporal aggregation
and single-image processing.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .utils import (
    BoundingBox, DetectedLine, FrameResult, AggregatedLine, TemporalResult,
    load_frames, save_frame_results, save_aggregated_results,
    save_single_image_result, draw_annotations
)
from .preprocessing import ImagePreprocessor
from .detection import LineDetector
from .recognition import OCREngine
from .postprocessing import FrameQualityAssessor, TemporalLineMatcher, TemporalAggregator


class TemporalOCRPipeline:
    """Main pipeline for temporal OCR processing."""

    def __init__(
        self,
        engine: str = "paddle",
        max_frames: int = 15,
        min_frames: int = 5,
        use_consensus: bool = True,
        blur_threshold: float = 100.0,
        padding: int = 12,
        lang: str = "en",
        adaptive_thresh: bool = False,
        model_path: str = None,
        ensemble_base_model: str = "microsoft/trocr-large-handwritten",
        ensemble_weights: Tuple[float, float] = (0.6, 0.4)
    ):
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.padding = padding

        # Initialize components
        self.ocr_engine = OCREngine(
            engine, lang, model_path=model_path,
            ensemble_base_model=ensemble_base_model,
            ensemble_weights=ensemble_weights
        )
        self.preprocessor = ImagePreprocessor(adaptive_thresh=adaptive_thresh)
        self.line_detector = LineDetector(self.ocr_engine, padding=padding)
        self.quality_assessor = FrameQualityAssessor(blur_threshold=blur_threshold)
        self.line_matcher = TemporalLineMatcher()
        self.aggregator = TemporalAggregator(use_consensus=use_consensus)

    def process(
        self,
        input_path: str,
        output_dir: str
    ) -> TemporalResult:
        """
        Process video or image folder.

        Args:
            input_path: Path to video file or folder of images
            output_dir: Directory for outputs

        Returns:
            TemporalResult with aggregated OCR results
        """
        # Setup output directories
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "crops").mkdir(exist_ok=True)
        (out_path / "annotated").mkdir(exist_ok=True)

        # Load frames
        frames = load_frames(input_path)
        print(f"[Pipeline] Loaded {len(frames)} frames")

        if not frames:
            print("[Pipeline] No frames loaded!")
            return TemporalResult(full_text="", frames_processed=0, frames_used=0)

        # Get image dimensions from first frame
        image_height = frames[0][1].shape[0]

        # Process each frame
        frame_results = []
        for frame_idx, image, frame_path in tqdm(frames, desc="Processing frames"):
            result = self._process_frame(
                frame_idx, image, frame_path, out_path
            )
            frame_results.append(result)

        # Select best frames
        selected_frames = self.quality_assessor.select_best_frames(
            frame_results, self.max_frames, self.min_frames
        )
        print(f"[Pipeline] Selected {len(selected_frames)} quality frames")

        if not selected_frames:
            print("[Pipeline] No valid frames after quality filtering!")
            return TemporalResult(
                full_text="",
                frames_processed=len(frame_results),
                frames_used=0
            )

        # Match lines across frames
        matched_lines = self.line_matcher.match_lines_across_frames(
            selected_frames, image_height
        )
        print(f"[Pipeline] Matched {len(matched_lines)} unique lines")

        # Aggregate results
        aggregated_lines = self.aggregator.aggregate_lines(matched_lines)

        # Build final result
        full_text = "\n".join(line.final_text for line in aggregated_lines)

        result = TemporalResult(
            full_text=full_text,
            lines=aggregated_lines,
            frames_processed=len(frame_results),
            frames_used=len(selected_frames),
            aggregation_method="consensus" if self.aggregator.use_consensus else "best_of_n"
        )

        # Save outputs
        self._save_outputs(result, frame_results, out_path)

        return result

    def _process_frame(
        self,
        frame_idx: int,
        image: np.ndarray,
        frame_path: str,
        out_path: Path
    ) -> FrameResult:
        """Process a single frame."""
        # Preprocess
        preprocessed = self.preprocessor.preprocess(image)

        # Estimate and correct skew
        skew_angle = self.preprocessor.estimate_skew(image)
        if abs(skew_angle) > 1.0:
            image = self.preprocessor.deskew(image, skew_angle)
            preprocessed = self.preprocessor.preprocess(image)

        # Detect lines
        lines = self.line_detector.detect_lines(image, preprocessed)

        # Calculate average OCR confidence
        avg_conf = 1.0
        if lines:
            all_confs = []
            for _, words in lines:
                all_confs.extend([w[2] for w in words])
            if all_confs:
                avg_conf = np.mean(all_confs)

        # Assess quality
        quality_score, is_valid, reason = self.quality_assessor.assess(
            image, len(lines), avg_conf
        )

        if not is_valid:
            return FrameResult(
                frame_index=frame_idx,
                frame_path=frame_path,
                quality_score=quality_score,
                lines=[],
                skew_angle=skew_angle,
                is_valid=False
            )

        # Process each line
        detected_lines = []
        frame_crop_dir = out_path / "crops" / f"frame_{frame_idx:04d}"
        frame_crop_dir.mkdir(parents=True, exist_ok=True)

        for line_idx, (line_bbox, words) in enumerate(lines):
            # Crop line
            line_crop = self.line_detector.crop_line(image, line_bbox)

            # Save crop
            crop_path = str(frame_crop_dir / f"line_{line_idx:02d}.png")
            cv2.imwrite(crop_path, line_crop)

            # OCR on crop (or use already detected text)
            if words:
                # Combine word texts
                line_text = " ".join(w[1] for w in words)
                line_conf = np.mean([w[2] for w in words])
            else:
                # Run OCR on crop
                line_text, line_conf = self.ocr_engine.recognize_line(line_crop)

            detected_lines.append(DetectedLine(
                index=line_idx,
                bbox=line_bbox,
                text=line_text.strip(),
                confidence=line_conf,
                crop_path=crop_path
            ))

        # Save annotated image
        annotated = draw_annotations(image, detected_lines)
        annotated_path = out_path / "annotated" / f"frame_{frame_idx:04d}.png"
        cv2.imwrite(str(annotated_path), annotated)

        return FrameResult(
            frame_index=frame_idx,
            frame_path=frame_path,
            quality_score=quality_score,
            lines=detected_lines,
            skew_angle=skew_angle,
            is_valid=True
        )

    def _save_outputs(
        self,
        result: TemporalResult,
        frame_results: List[FrameResult],
        out_path: Path
    ):
        """Save JSON outputs."""
        save_frame_results(frame_results, out_path)
        save_aggregated_results(result, out_path)
        print(f"\n[Output] Saved results to {out_path}")


class LineCropOCRPipeline:
    """
    Single-image pipeline that crops lines before OCR.
    Handles focus gradients, exposure issues, and perspective.

    Vision verification uses a local LLM (LLaVA) to verify and correct
    TrOCR recognition errors using semantic understanding.
    """

    def __init__(
        self,
        engine: str = "paddle",
        padding: int = 12,
        lang: str = "en",
        adaptive_thresh: bool = False,
        deskew: bool = True,
        model_path: str = None,
        vision_verify: bool = False,
        vision_model: str = "llava:7b",
        vision_mode: str = "verify_all",
        ensemble_base_model: str = "microsoft/trocr-large-handwritten",
        ensemble_weights: Tuple[float, float] = (0.6, 0.4)
    ):
        self.padding = padding
        self.deskew = deskew

        self.ocr_engine = OCREngine(
            engine, lang, model_path=model_path,
            ensemble_base_model=ensemble_base_model,
            ensemble_weights=ensemble_weights
        )
        self.preprocessor = ImagePreprocessor(adaptive_thresh=adaptive_thresh)
        self.line_detector = LineDetector(self.ocr_engine, padding=padding)

        # Vision verification (optional but recommended)
        self.vision_verifier = None
        if vision_verify:
            try:
                from vision_verification import create_verifier
                self.vision_verifier = create_verifier(
                    verifier_type="ollama",
                    model=vision_model,
                    mode=vision_mode
                )
            except Exception as e:
                print(f"[Warning] Vision verification disabled: {e}")

        # Strikethrough detection (always enabled)
        try:
            from strikethrough_filter import StrikethroughDetector, filter_strikethrough_text
            self.strikethrough_detector = StrikethroughDetector()
            self.filter_strikethrough = filter_strikethrough_text
            print("[OCR] Strikethrough detection enabled")
        except Exception as e:
            self.strikethrough_detector = None
            self.filter_strikethrough = None
            print(f"[Warning] Strikethrough detection disabled: {e}")

    def process_image(
        self,
        image_path: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """Process a single image."""
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Could not load image: {image_path}"}

        return self._process_single(image, image_path, output_dir)

    def process_folder(
        self,
        folder_path: str,
        output_dir: str
    ) -> List[Dict[str, Any]]:
        """Process all images in a folder."""
        folder = Path(folder_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        image_files = sorted([
            f for f in folder.iterdir()
            if f.suffix.lower() in image_extensions
        ])

        results = []
        for img_path in tqdm(image_files, desc="Processing images"):
            result = self.process_image(str(img_path), output_dir)
            results.append(result)

        return results

    def _process_single(
        self,
        image: np.ndarray,
        image_path: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """Process a single image."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        image_name = Path(image_path).stem

        # Preprocess
        preprocessed = self.preprocessor.preprocess(image)

        # Deskew if enabled
        skew_angle = 0.0
        if self.deskew:
            skew_angle = self.preprocessor.estimate_skew(image)
            if abs(skew_angle) > 1.0:
                image = self.preprocessor.deskew(image, skew_angle)
                preprocessed = self.preprocessor.preprocess(image)

        # Detect lines
        lines = self.line_detector.detect_lines(image, preprocessed)

        # Setup crop directory
        crop_dir = out_path / "crops" / image_name
        crop_dir.mkdir(parents=True, exist_ok=True)

        # Process each line
        line_results = []
        for line_idx, (line_bbox, words) in enumerate(lines):
            # Crop
            line_crop = self.line_detector.crop_line(image, line_bbox)

            # Save crop
            crop_path = crop_dir / f"line_{line_idx:02d}.png"
            cv2.imwrite(str(crop_path), line_crop)

            # Get text from TrOCR
            if words:
                line_text = " ".join(w[1] for w in words)
                line_conf = np.mean([w[2] for w in words])
            else:
                line_text, line_conf = self.ocr_engine.recognize_line(line_crop)

            # Vision verification: use local LLM to verify/correct TrOCR output
            vision_text = None
            if self.vision_verifier:
                result = self.vision_verifier.verify_recognition(
                    image=line_crop,
                    trocr_text=line_text,
                    trocr_confidence=line_conf
                )
                line_text = result.verified_text
                line_conf = result.verified_confidence
                vision_text = result.vision_text

            # Strikethrough filtering: remove crossed-out text
            had_strikethrough = False
            if self.strikethrough_detector and self.filter_strikethrough:
                filtered_text, had_strikethrough, strike_ratio = self.filter_strikethrough(
                    line_crop, line_text, self.strikethrough_detector
                )
                if had_strikethrough:
                    print(f"[Strikethrough] '{line_text}' -> '{filtered_text}' (ratio: {strike_ratio:.1%})")
                    line_text = filtered_text

            line_results.append({
                "index": line_idx,
                "bbox": [line_bbox.x, line_bbox.y, line_bbox.width, line_bbox.height],
                "text": line_text.strip(),
                "confidence": line_conf,
                "crop_path": str(crop_path)
            })

        # Build full text
        full_text = "\n".join(lr["text"] for lr in line_results)

        # Save annotated image
        annotated = image.copy()
        for lr in line_results:
            x, y, w, h = lr["bbox"]
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"L{lr['index']}"
            cv2.putText(annotated, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        annotated_path = out_path / "annotated" / f"{image_name}.png"
        annotated_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(annotated_path), annotated)

        result = {
            "image_path": image_path,
            "full_text": full_text,
            "skew_angle": skew_angle,
            "lines": line_results
        }

        # Save JSON
        save_single_image_result(result, out_path, image_name)

        return result
