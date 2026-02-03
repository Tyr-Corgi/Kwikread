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
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import warnings

import cv2
import numpy as np
from tqdm import tqdm

# Suppress specific warnings from dependencies (instead of blanket suppression)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BoundingBox:
    """Represents a bounding box for a text region."""
    x: int
    y: int
    width: int
    height: int

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def area(self) -> int:
        return self.width * self.height

    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another box."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def y_overlap_ratio(self, other: 'BoundingBox') -> float:
        """Calculate vertical overlap ratio."""
        y1 = max(self.y, other.y)
        y2 = min(self.y + self.height, other.y + other.height)

        if y2 <= y1:
            return 0.0

        overlap = y2 - y1
        min_height = min(self.height, other.height)
        return overlap / min_height if min_height > 0 else 0.0


@dataclass
class DetectedLine:
    """Represents a detected text line in a frame."""
    index: int
    bbox: BoundingBox
    text: str
    confidence: float
    crop_path: Optional[str] = None


@dataclass
class FrameResult:
    """Results from processing a single frame."""
    frame_index: int
    frame_path: str
    quality_score: float
    lines: List[DetectedLine] = field(default_factory=list)
    skew_angle: float = 0.0
    is_valid: bool = True


@dataclass
class AggregatedLine:
    """Aggregated result for a single line across frames."""
    line_index: int
    final_text: str
    confidence: float
    consensus_text: str
    best_single_text: str
    best_single_confidence: float
    contributing_frames: List[int] = field(default_factory=list)
    all_candidates: List[Tuple[str, float, int]] = field(default_factory=list)


@dataclass
class TemporalResult:
    """Final aggregated results across all frames."""
    full_text: str
    lines: List[AggregatedLine] = field(default_factory=list)
    frames_processed: int = 0
    frames_used: int = 0
    aggregation_method: str = "consensus"


# =============================================================================
# OCR Engine Wrapper
# =============================================================================

class OCREngine:
    """Wrapper for OCR engines (PaddleOCR, EasyOCR, Tesseract, TrOCR)."""

    def __init__(self, engine_name: str = "paddle", lang: str = "en", model_path: str = None):
        self.engine_name = engine_name.lower()
        self.lang = lang
        self.engine = None
        self.model_path = model_path
        self.trocr_model = None
        self.trocr_processor = None
        self.detector = None  # For TrOCR, we need a separate detector
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the selected OCR engine."""
        if self.engine_name == "paddle":
            try:
                from paddleocr import PaddleOCR
                # use_angle_cls for rotated text, det for detection, rec for recognition
                self.engine = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.lang,
                    use_gpu=False,
                    show_log=False,
                    det_db_thresh=0.3,
                    det_db_box_thresh=0.5,
                    rec_batch_num=6
                )
                print(f"[OCR] Initialized PaddleOCR (lang={self.lang})")
            except ImportError:
                print("[OCR] PaddleOCR not available, falling back to EasyOCR")
                self.engine_name = "easyocr"
                self._initialize_engine()

        elif self.engine_name == "easyocr":
            try:
                import easyocr
                self.engine = easyocr.Reader([self.lang], gpu=False, verbose=False)
                print(f"[OCR] Initialized EasyOCR (lang={self.lang})")
            except ImportError:
                print("[OCR] EasyOCR not available, falling back to Tesseract")
                self.engine_name = "tesseract"
                self._initialize_engine()

        elif self.engine_name == "tesseract":
            try:
                import pytesseract
                self.engine = pytesseract
                print("[OCR] Initialized Tesseract (warning: weak for handwriting)")
            except ImportError:
                raise RuntimeError("No OCR engine available. Install paddleocr or easyocr.")

        elif self.engine_name == "trocr":
            try:
                import torch
                from transformers import VisionEncoderDecoderModel, TrOCRProcessor
                from PIL import Image

                # Try large model for better accuracy on challenging handwriting
                model_path = self.model_path or "microsoft/trocr-large-handwritten"

                print(f"[OCR] Loading TrOCR from: {model_path}")
                self.trocr_processor = TrOCRProcessor.from_pretrained(model_path)
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_path)
                self.trocr_model.eval()

                # Use MPS on Mac if available, else CPU
                if torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                    print("[OCR] Using MPS (Metal) acceleration")
                else:
                    self.device = torch.device("cpu")
                self.trocr_model = self.trocr_model.to(self.device)

                # Initialize EasyOCR as detector (TrOCR only does recognition)
                import easyocr
                self.detector = easyocr.Reader([self.lang], gpu=False, verbose=False)
                print(f"[OCR] Initialized TrOCR with EasyOCR detector")

            except ImportError as e:
                print(f"[OCR] TrOCR not available ({e}), falling back to EasyOCR")
                self.engine_name = "easyocr"
                self._initialize_engine()

        elif self.engine_name == "claude":
            try:
                import anthropic
                import base64

                self.claude_client = anthropic.Anthropic()

                # Initialize EasyOCR as detector (Claude does recognition)
                import easyocr
                self.detector = easyocr.Reader([self.lang], gpu=False, verbose=False)
                print(f"[OCR] Initialized Claude Vision with EasyOCR detector")

            except ImportError as e:
                print(f"[OCR] Claude SDK not available ({e}), falling back to EasyOCR")
                self.engine_name = "easyocr"
                self._initialize_engine()

    def detect_and_recognize(self, image: np.ndarray) -> List[Tuple[List[List[int]], str, float]]:
        """
        Detect text regions and recognize text.

        Returns:
            List of (polygon_points, text, confidence)
        """
        if image is None or image.size == 0:
            return []

        results = []

        if self.engine_name == "paddle":
            try:
                ocr_result = self.engine.ocr(image, cls=True)
                if ocr_result and ocr_result[0]:
                    for item in ocr_result[0]:
                        if item is None:
                            continue
                        polygon = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        text, conf = item[1]
                        results.append((polygon, text, conf))
            except Exception as e:
                print(f"[OCR] PaddleOCR error: {e}")

        elif self.engine_name == "easyocr":
            try:
                # Word-level detection to handle multi-column layouts
                ocr_result = self.engine.readtext(
                    image,
                    paragraph=False,
                    width_ths=0.1,  # Prevent horizontal word merging
                    height_ths=0.5
                )
                for item in ocr_result:
                    polygon = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text = item[1]
                    conf = item[2]
                    results.append((polygon, text, conf))
            except Exception as e:
                print(f"[OCR] EasyOCR error: {e}")

        elif self.engine_name == "tesseract":
            try:
                data = self.engine.image_to_data(image, output_type=self.engine.Output.DICT)
                for i, text in enumerate(data['text']):
                    if text.strip():
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        conf = float(data['conf'][i]) / 100.0 if data['conf'][i] != -1 else 0.5
                        polygon = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                        results.append((polygon, text, conf))
            except Exception as e:
                print(f"[OCR] Tesseract error: {e}")

        elif self.engine_name == "trocr":
            # Use EasyOCR for detection, TrOCR for recognition
            try:
                # Get detections - use moderate merging, we'll split wide ones ourselves
                detections = self.detector.readtext(
                    image,
                    paragraph=False,
                    width_ths=0.3,
                    height_ths=0.5
                )

                # Split wide detections using connected component analysis
                final_detections = []
                for item in detections:
                    polygon = item[0]
                    x_coords = [p[0] for p in polygon]
                    y_coords = [p[1] for p in polygon]
                    x1, x2 = int(min(x_coords)), int(max(x_coords))
                    y1, y2 = int(min(y_coords)), int(max(y_coords))
                    width = x2 - x1
                    height = y2 - y1

                    # Check if this detection is wide enough to potentially span columns
                    if width > 400 and width > height * 3:
                        # Extract crop and find text blobs
                        pad = 3
                        crop = image[max(0,y1-pad):min(image.shape[0],y2+pad),
                                    max(0,x1-pad):min(image.shape[1],x2+pad)]

                        if crop.size > 0:
                            # Convert to grayscale and binarize
                            if len(crop.shape) == 3:
                                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            else:
                                gray = crop.copy()

                            # Use Otsu threshold (better for this image)
                            _, binary = cv2.threshold(gray, 0, 255,
                                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                            # Dilate to connect nearby text parts
                            kernel = np.ones((3, 3), np.uint8)
                            binary = cv2.dilate(binary, kernel, iterations=2)

                            # Find connected components (text blobs)
                            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                                binary, connectivity=8
                            )

                            # Get bounding boxes of significant blobs
                            blobs = []
                            for i in range(1, num_labels):
                                area = stats[i, cv2.CC_STAT_AREA]
                                if area > 100:  # Filter noise
                                    bx = stats[i, cv2.CC_STAT_LEFT]
                                    bw = stats[i, cv2.CC_STAT_WIDTH]
                                    blobs.append((bx, bx + bw))  # (left, right)

                            if len(blobs) >= 2:
                                # Sort by left edge
                                blobs.sort(key=lambda b: b[0])

                                # Find gaps between blobs
                                max_gap = 0
                                split_x_local = None
                                for i in range(len(blobs) - 1):
                                    # Gap = start of next blob - end of current blob
                                    gap = blobs[i+1][0] - blobs[i][1]
                                    if gap > max_gap:
                                        max_gap = gap
                                        split_x_local = (blobs[i][1] + blobs[i+1][0]) / 2

                                # If gap is significant (>80px), split
                                if max_gap > 80 and split_x_local is not None:
                                    split_x = x1 + int(split_x_local)

                                    # Create two polygons
                                    poly1 = [[x1, y1], [split_x-5, y1], [split_x-5, y2], [x1, y2]]
                                    poly2 = [[split_x+5, y1], [x2, y1], [x2, y2], [split_x+5, y2]]
                                    final_detections.append((poly1, "", 0))
                                    final_detections.append((poly2, "", 0))
                                    continue

                    final_detections.append(item)

                # Process each detection with TrOCR
                for item in final_detections:
                    polygon = item[0]
                    x_coords = [p[0] for p in polygon]
                    y_coords = [p[1] for p in polygon]
                    x1, x2 = int(min(x_coords)), int(max(x_coords))
                    y1, y2 = int(min(y_coords)), int(max(y_coords))

                    # Pad slightly
                    pad = 5
                    x1, y1 = max(0, x1-pad), max(0, y1-pad)
                    x2, y2 = min(image.shape[1], x2+pad), min(image.shape[0], y2+pad)

                    crop = image[y1:y2, x1:x2]
                    if crop.size > 0:
                        text, conf = self.recognize_line(crop)
                        if text:
                            results.append((polygon, text, conf))
            except Exception as e:
                print(f"[OCR] TrOCR detection error: {e}")

        elif self.engine_name == "claude":
            # Use EasyOCR for detection, Claude for recognition
            try:
                # Word-level detection to handle multi-column layouts
                detections = self.detector.readtext(
                    image,
                    paragraph=False,
                    width_ths=0.1,
                    height_ths=0.5
                )
                for item in detections:
                    polygon = item[0]
                    x_coords = [p[0] for p in polygon]
                    y_coords = [p[1] for p in polygon]
                    x1, x2 = int(min(x_coords)), int(max(x_coords))
                    y1, y2 = int(min(y_coords)), int(max(y_coords))

                    pad = 5
                    x1, y1 = max(0, x1-pad), max(0, y1-pad)
                    x2, y2 = min(image.shape[1], x2+pad), min(image.shape[0], y2+pad)

                    crop = image[y1:y2, x1:x2]
                    if crop.size > 0:
                        text, conf = self.recognize_line(crop)
                        if text:
                            results.append((polygon, text, conf))
            except Exception as e:
                print(f"[OCR] Claude detection error: {e}")

        return results

    def recognize_line(self, line_image: np.ndarray) -> Tuple[str, float]:
        """
        Recognize text in a cropped line image.

        Returns:
            (text, confidence)
        """
        if line_image is None or line_image.size == 0:
            return "", 0.0

        if self.engine_name == "paddle":
            try:
                result = self.engine.ocr(line_image, det=False, cls=True)
                if result and result[0]:
                    # Combine all recognized text parts
                    texts = []
                    confs = []
                    for item in result[0]:
                        if item:
                            texts.append(item[0])
                            confs.append(item[1])
                    if texts:
                        return " ".join(texts), sum(confs) / len(confs)
            except Exception as e:
                print(f"[OCR] PaddleOCR recognize_line error: {e}")

        elif self.engine_name == "easyocr":
            try:
                result = self.engine.readtext(line_image)
                if result:
                    texts = [item[1] for item in result]
                    confs = [item[2] for item in result]
                    return " ".join(texts), sum(confs) / len(confs) if confs else 0.0
            except Exception as e:
                print(f"[OCR] EasyOCR recognize_line error: {e}")

        elif self.engine_name == "tesseract":
            try:
                text = self.engine.image_to_string(line_image).strip()
                return text, 0.5  # Tesseract doesn't give confidence easily
            except Exception as e:
                print(f"[OCR] Tesseract recognize_line error: {e}")

        elif self.engine_name == "trocr":
            try:
                import torch
                from PIL import Image, ImageOps, ImageEnhance, ImageFilter

                # === GENTLE PREPROCESSING FOR HANDWRITING (preserve strokes) ===

                # MINIMAL PREPROCESSING - TrOCR may work better with natural images

                # Step 1: Convert to PIL Image directly (keep color for now)
                if len(line_image.shape) == 3:
                    pil_orig = Image.fromarray(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
                else:
                    pil_orig = Image.fromarray(line_image).convert("RGB")

                # Step 2: Add padding
                pad = 16
                w, h = pil_orig.size
                padded = Image.new("RGB", (w + 2*pad, h + 2*pad), (255, 255, 255))
                padded.paste(pil_orig, (pad, pad))

                # Step 3: Resize to reasonable height
                w, h = padded.size
                target_height = 96
                scale = target_height / h
                new_w = max(int(w * scale), 64)
                resized = padded.resize((new_w, target_height), Image.LANCZOS)

                # Step 4: Convert to grayscale and enhance contrast
                gray_pil = resized.convert("L")

                # Auto-contrast (stretches histogram)
                gray_pil = ImageOps.autocontrast(gray_pil, cutoff=2)

                # Convert back to RGB
                pil_image = gray_pil.convert("RGB")

                # Boost contrast
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.3)

                # Process and generate with optimized parameters
                pixel_values = self.trocr_processor(
                    pil_image, return_tensors="pt"
                ).pixel_values.to(self.device)

                with torch.no_grad():
                    generated_ids = self.trocr_model.generate(
                        pixel_values,
                        max_length=64,
                        num_beams=5,  # More beams for better decoding
                        early_stopping=True,
                        length_penalty=1.0,
                        no_repeat_ngram_size=3,  # Prevent repetition
                    )

                text = self.trocr_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()

                # POST-PROCESSING: Clean up TrOCR word output
                # Remove spurious periods/punctuation (common TrOCR artifact)
                text = re.sub(r'\s*[.,;:]\s*', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                # Line-level corrections happen after words are grouped

                # TrOCR doesn't provide confidence directly; estimate based on output
                conf = 0.9
                if not text or len(text) < 2:
                    conf = 0.1
                elif any(x in text.lower() for x in ['%', 'na%', 'sc%', '###']):
                    conf = 0.1  # Garbage output

                return text, conf

            except Exception as e:
                print(f"[OCR] TrOCR recognition error: {e}")
                import traceback
                traceback.print_exc()

        return "", 0.0


# =============================================================================
# Image Preprocessing
# =============================================================================

class ImagePreprocessor:
    """Preprocessing pipeline optimized for handwriting OCR."""

    def __init__(
        self,
        clahe_clip: float = 2.0,
        clahe_grid: int = 8,
        denoise_strength: int = 7,
        adaptive_thresh: bool = False,
        morph_kernel_size: int = 2
    ):
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.denoise_strength = denoise_strength
        self.adaptive_thresh = adaptive_thresh
        self.morph_kernel_size = morph_kernel_size

    def preprocess(self, image: np.ndarray, for_ocr: bool = True) -> np.ndarray:
        """
        Full preprocessing pipeline.

        Args:
            image: Input BGR image
            for_ocr: If True, optimize for OCR; if False, optimize for detection

        Returns:
            Preprocessed image
        """
        if image is None:
            return None

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # White balance / illumination normalization
        gray = self._normalize_illumination(gray)

        # CLAHE contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=(self.clahe_grid, self.clahe_grid)
        )
        enhanced = clahe.apply(gray)

        # Light denoising (bilateral preserves edges)
        if self.denoise_strength > 0:
            denoised = cv2.bilateralFilter(
                enhanced,
                d=self.denoise_strength,
                sigmaColor=75,
                sigmaSpace=75
            )
        else:
            denoised = enhanced

        # Optional adaptive thresholding for very low contrast
        if self.adaptive_thresh and for_ocr:
            binary = cv2.adaptiveThreshold(
                denoised, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=2
            )
            # Light morphology to connect broken strokes
            if self.morph_kernel_size > 0:
                kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            return binary

        return denoised

    def _normalize_illumination(self, gray: np.ndarray) -> np.ndarray:
        """Remove uneven illumination / shadows."""
        # Estimate background using large blur
        blur = cv2.GaussianBlur(gray, (51, 51), 0)
        # Divide to normalize
        normalized = cv2.divide(gray, blur, scale=255)
        return normalized.astype(np.uint8)

    def estimate_skew(self, image: np.ndarray) -> float:
        """
        Estimate skew angle of text using Hough transform.

        Returns:
            Skew angle in degrees
        """
        if image is None:
            return 0.0

        # Get edges
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            threshold=100,
            minLineLength=50,
            maxLineGap=10
        )

        if lines is None or len(lines) == 0:
            return 0.0

        # Calculate angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only consider near-horizontal lines
                if abs(angle) < 45:
                    angles.append(angle)

        if not angles:
            return 0.0

        # Median angle
        return np.median(angles)

    def deskew(self, image: np.ndarray, angle: float = None) -> np.ndarray:
        """
        Deskew image by rotating.

        Args:
            image: Input image
            angle: Rotation angle (if None, estimate automatically)

        Returns:
            Deskewed image
        """
        if image is None:
            return None

        if angle is None:
            angle = self.estimate_skew(image)

        if abs(angle) < 0.5:  # Skip if nearly straight
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding box
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust rotation matrix
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Rotate
        rotated = cv2.warpAffine(
            image, M, (new_w, new_h),
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated


# =============================================================================
# Line Detection and Grouping
# =============================================================================

class LineDetector:
    """Detects and groups text into lines."""

    def __init__(
        self,
        ocr_engine: OCREngine,
        y_overlap_thresh: float = 0.5,
        min_line_height: int = 10,
        padding: int = 12
    ):
        self.ocr_engine = ocr_engine
        self.y_overlap_thresh = y_overlap_thresh
        self.min_line_height = min_line_height
        self.padding = padding

    def detect_lines(
        self,
        image: np.ndarray,
        preprocessed: np.ndarray = None
    ) -> List[Tuple[BoundingBox, List[Tuple[BoundingBox, str, float]]]]:
        """
        Detect text and group into lines.

        Returns:
            List of (line_bbox, [(word_bbox, text, conf), ...])
        """
        # Use original image for detection (OCR engines handle preprocessing)
        detections = self.ocr_engine.detect_and_recognize(image)

        if not detections:
            return []

        # Convert polygons to bounding boxes
        word_boxes = []
        for polygon, text, conf in detections:
            if not text.strip():
                continue

            # Convert polygon to axis-aligned bbox
            pts = np.array(polygon)
            x = int(pts[:, 0].min())
            y = int(pts[:, 1].min())
            w = int(pts[:, 0].max() - x)
            h = int(pts[:, 1].max() - y)

            if h < self.min_line_height:
                continue

            bbox = BoundingBox(x, y, w, h)
            word_boxes.append((bbox, text, conf))

        if not word_boxes:
            return []

        # Group into lines by y-overlap clustering
        lines = self._group_into_lines(word_boxes)

        return lines

    def _group_into_lines(
        self,
        word_boxes: List[Tuple[BoundingBox, str, float]]
    ) -> List[Tuple[BoundingBox, List[Tuple[BoundingBox, str, float]]]]:
        """
        Group word boxes into lines with COLUMN-AWARE detection.
        Handles multi-column layouts and prevents merging items across columns.
        Also handles diagonal text by treating isolated words as separate items.
        """
        if not word_boxes:
            return []

        from sklearn.cluster import DBSCAN

        # Estimate typical dimensions
        heights = [box.height for box, _, _ in word_boxes]
        widths = [box.width for box, _, _ in word_boxes]
        median_height = np.median(heights)
        median_width = np.median(widths)

        # STEP 1: First cluster by Y to find rough horizontal bands
        y_centers = np.array([[box.center_y] for box, _, _ in word_boxes])
        y_eps = median_height * 0.7
        y_clustering = DBSCAN(eps=max(y_eps, 15), min_samples=1).fit(y_centers)

        # STEP 2: Within each Y-cluster, split by large X gaps (column detection)
        # This prevents merging items from different columns
        # Use minimum of median_width * 1.5 OR 100px - whichever is smaller
        # This catches column gaps which are typically >100px
        MAX_WORD_GAP = min(median_width * 1.5, 100)  # Max gap between words in same item

        final_lines = []
        y_groups = {}
        for i, label in enumerate(y_clustering.labels_):
            if label not in y_groups:
                y_groups[label] = []
            y_groups[label].append(word_boxes[i])

        for label, words in y_groups.items():
            # Sort words left-to-right
            words = sorted(words, key=lambda x: x[0].x)

            # Split into separate items based on X gaps
            current_item = [words[0]]
            for i in range(1, len(words)):
                prev_box = words[i-1][0]
                curr_box = words[i][0]

                # Calculate gap between end of previous word and start of current
                gap = curr_box.x - (prev_box.x + prev_box.width)

                if gap > MAX_WORD_GAP:
                    # Large gap - this is a new column/item
                    if current_item:
                        final_lines.append(current_item)
                    current_item = [words[i]]
                else:
                    # Same item, add to current group
                    current_item.append(words[i])

            if current_item:
                final_lines.append(current_item)

        # STEP 3: Create line bounding boxes for each item
        lines = []
        for words in final_lines:
            min_x = min(box.x for box, _, _ in words)
            min_y = min(box.y for box, _, _ in words)
            max_x = max(box.x + box.width for box, _, _ in words)
            max_y = max(box.y + box.height for box, _, _ in words)

            line_bbox = BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)
            lines.append((line_bbox, words))

        # Sort lines by Y first (top-to-bottom), then X (left-to-right)
        # This gives natural reading order for multi-column layouts
        lines = sorted(lines, key=lambda x: (x[0].center_y // 50, x[0].x))

        return lines

    def crop_line(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        padding: int = None
    ) -> np.ndarray:
        """Crop a line region with padding."""
        if padding is None:
            padding = self.padding

        h, w = image.shape[:2]

        x1 = max(0, bbox.x - padding)
        y1 = max(0, bbox.y - padding)
        x2 = min(w, bbox.x + bbox.width + padding)
        y2 = min(h, bbox.y + bbox.height + padding)

        return image[y1:y2, x1:x2].copy()


# =============================================================================
# Frame Quality Assessment
# =============================================================================

class FrameQualityAssessor:
    """Assesses frame quality for filtering."""

    def __init__(
        self,
        blur_threshold: float = 100.0,
        brightness_min: int = 30,
        brightness_max: int = 225,
        min_text_regions: int = 1
    ):
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.min_text_regions = min_text_regions

    def assess(
        self,
        image: np.ndarray,
        num_detections: int = 0,
        avg_confidence: float = 1.0
    ) -> Tuple[float, bool, str]:
        """
        Assess frame quality.

        Returns:
            (quality_score, is_valid, rejection_reason)
        """
        if image is None:
            return 0.0, False, "null_image"

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Blur metric (variance of Laplacian)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()

        if blur_score < self.blur_threshold:
            return blur_score, False, f"too_blurry ({blur_score:.1f} < {self.blur_threshold})"

        # Brightness check
        mean_brightness = gray.mean()
        if mean_brightness < self.brightness_min:
            return blur_score, False, f"too_dark ({mean_brightness:.1f})"
        if mean_brightness > self.brightness_max:
            return blur_score, False, f"too_bright ({mean_brightness:.1f})"

        # Text presence check
        if num_detections < self.min_text_regions:
            return blur_score, False, f"no_text_detected ({num_detections} regions)"

        # OCR confidence check - reject frames with garbage OCR
        if avg_confidence < 0.3:
            return blur_score, False, f"low_ocr_confidence ({avg_confidence:.2f})"

        # Quality score combines blur and brightness stability
        brightness_penalty = abs(mean_brightness - 128) / 128  # Prefer mid-gray
        quality_score = blur_score * (1 - 0.3 * brightness_penalty) * avg_confidence

        return quality_score, True, "ok"

    def select_best_frames(
        self,
        frame_results: List[FrameResult],
        max_frames: int = 15,
        min_frames: int = 5
    ) -> List[FrameResult]:
        """Select the best frames based on quality score."""
        # Filter valid frames
        valid = [f for f in frame_results if f.is_valid and len(f.lines) > 0]

        if len(valid) <= max_frames:
            return valid

        # Sort by quality and select top N
        valid = sorted(valid, key=lambda x: x.quality_score, reverse=True)
        return valid[:max_frames]


# =============================================================================
# Temporal Line Matching
# =============================================================================

class TemporalLineMatcher:
    """Matches lines across frames for temporal aggregation."""

    def __init__(
        self,
        y_tolerance: float = 0.20,  # 20% of image height (more permissive)
        iou_threshold: float = 0.3,
        text_similarity_weight: float = 0.4,  # Weight text more for matching
        merge_similar_threshold: float = 0.7  # Merge lines with >70% text similarity
    ):
        self.y_tolerance = y_tolerance
        self.iou_threshold = iou_threshold
        self.text_similarity_weight = text_similarity_weight
        self.merge_similar_threshold = merge_similar_threshold

    def match_lines_across_frames(
        self,
        frame_results: List[FrameResult],
        image_height: int
    ) -> Dict[int, List[Tuple[int, DetectedLine]]]:
        """
        Match lines across frames.

        Returns:
            {canonical_line_index: [(frame_index, DetectedLine), ...]}
        """
        if not frame_results:
            return {}

        # Use first valid frame as reference
        reference_frame = None
        for fr in frame_results:
            if fr.lines:
                reference_frame = fr
                break

        if reference_frame is None:
            return {}

        # Initialize canonical lines from reference
        canonical_lines = {
            i: [(reference_frame.frame_index, line)]
            for i, line in enumerate(reference_frame.lines)
        }

        y_tol = image_height * self.y_tolerance

        # Match other frames to canonical lines
        for fr in frame_results:
            if fr.frame_index == reference_frame.frame_index:
                continue

            for line in fr.lines:
                best_match = self._find_best_match(
                    line, canonical_lines, y_tol, image_height
                )

                if best_match is not None:
                    canonical_lines[best_match].append((fr.frame_index, line))
                else:
                    # New line not seen in reference
                    new_idx = len(canonical_lines)
                    canonical_lines[new_idx] = [(fr.frame_index, line)]

        # Re-sort canonical lines by average y-position
        sorted_lines = {}
        for old_idx, matches in canonical_lines.items():
            avg_y = np.mean([m[1].bbox.center_y for m in matches])
            sorted_lines[old_idx] = (avg_y, matches)

        sorted_items = sorted(sorted_lines.items(), key=lambda x: x[1][0])

        return {
            new_idx: matches
            for new_idx, (old_idx, (_, matches)) in enumerate(sorted_items)
        }

    def _find_best_match(
        self,
        line: DetectedLine,
        canonical_lines: Dict[int, List[Tuple[int, DetectedLine]]],
        y_tol: float,
        image_height: int
    ) -> Optional[int]:
        """Find the best matching canonical line."""
        best_idx = None
        best_score = 0.0

        for idx, matches in canonical_lines.items():
            # Use median y-position of canonical line
            canonical_y = np.median([m[1].bbox.center_y for m in matches])

            # Y-distance score
            y_dist = abs(line.bbox.center_y - canonical_y)
            if y_dist > y_tol:
                continue

            y_score = 1 - (y_dist / y_tol)

            # Text similarity score (using best match from canonical)
            text_scores = []
            for _, canonical_line in matches:
                sim = self._text_similarity(line.text, canonical_line.text)
                text_scores.append(sim)
            text_score = max(text_scores) if text_scores else 0

            # Combined score
            score = y_score * (1 - self.text_similarity_weight) + text_score * self.text_similarity_weight

            if score > best_score and score > 0.5:
                best_score = score
                best_idx = idx

        return best_idx

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using Levenshtein ratio."""
        if not text1 or not text2:
            return 0.0

        try:
            from rapidfuzz import fuzz
            return fuzz.ratio(text1.lower(), text2.lower()) / 100.0
        except ImportError:
            # Fallback to basic comparison
            return 1.0 if text1.lower() == text2.lower() else 0.0


# =============================================================================
# Temporal Aggregation
# =============================================================================

class TemporalAggregator:
    """Aggregates OCR results across frames."""

    def __init__(
        self,
        consensus_threshold: float = 0.6,
        min_agreement: int = 2,
        use_consensus: bool = True
    ):
        self.consensus_threshold = consensus_threshold
        self.min_agreement = min_agreement
        self.use_consensus = use_consensus

    def aggregate_lines(
        self,
        matched_lines: Dict[int, List[Tuple[int, DetectedLine]]]
    ) -> List[AggregatedLine]:
        """
        Aggregate matched lines into final results.

        Returns:
            List of AggregatedLine objects
        """
        aggregated = []

        for line_idx, matches in sorted(matched_lines.items()):
            if not matches:
                continue

            # Extract candidates
            candidates = [
                (line.text, line.confidence, frame_idx)
                for frame_idx, line in matches
            ]

            # Best-of-N (highest confidence)
            best_candidate = max(candidates, key=lambda x: x[1])
            best_text = best_candidate[0]
            best_conf = best_candidate[1]

            # Consensus voting
            if self.use_consensus and len(candidates) >= self.min_agreement:
                consensus_text, consensus_conf = self._consensus_vote(candidates)
            else:
                consensus_text = best_text
                consensus_conf = best_conf

            # Choose final result
            if consensus_conf >= self.consensus_threshold:
                final_text = consensus_text
                final_conf = consensus_conf
            else:
                final_text = best_text
                final_conf = best_conf

            aggregated.append(AggregatedLine(
                line_index=line_idx,
                final_text=final_text,
                confidence=final_conf,
                consensus_text=consensus_text,
                best_single_text=best_text,
                best_single_confidence=best_conf,
                contributing_frames=[frame_idx for frame_idx, _ in matches],
                all_candidates=candidates
            ))

        return aggregated

    def _consensus_vote(
        self,
        candidates: List[Tuple[str, float, int]]
    ) -> Tuple[str, float]:
        """
        Character-level consensus voting with alignment.

        Returns:
            (consensus_text, confidence)
        """
        if not candidates:
            return "", 0.0

        if len(candidates) == 1:
            return candidates[0][0], candidates[0][1]

        texts = [c[0] for c in candidates]
        confs = [c[1] for c in candidates]

        # Find reference (longest or highest confidence)
        ref_idx = max(range(len(texts)), key=lambda i: (len(texts[i]), confs[i]))
        reference = texts[ref_idx]

        if not reference:
            return "", 0.0

        # Align all texts to reference and vote character-by-character
        try:
            from rapidfuzz import process
            from rapidfuzz.distance import Levenshtein
        except ImportError:
            # Fallback: just return highest confidence
            best_idx = max(range(len(candidates)), key=lambda i: confs[i])
            return texts[best_idx], confs[best_idx]

        # Simple approach: for each position, vote on character
        # Use weighted voting by confidence
        aligned_texts = [self._align_to_reference(reference, t) for t in texts]

        consensus_chars = []
        for pos in range(len(reference)):
            char_votes = {}
            total_weight = 0

            for i, aligned in enumerate(aligned_texts):
                if pos < len(aligned):
                    char = aligned[pos]
                    weight = confs[i]
                    char_votes[char] = char_votes.get(char, 0) + weight
                    total_weight += weight

            if char_votes:
                best_char = max(char_votes.items(), key=lambda x: x[1])
                consensus_chars.append(best_char[0])

        consensus_text = "".join(consensus_chars)

        # Calculate consensus confidence
        # Higher if more agreement, weighted by individual confidences
        avg_conf = np.mean(confs)

        # Calculate agreement score
        similarities = []
        for t in texts:
            sim = Levenshtein.normalized_similarity(consensus_text, t)
            similarities.append(sim)
        agreement = np.mean(similarities)

        consensus_conf = avg_conf * agreement

        return consensus_text.strip(), consensus_conf

    def _align_to_reference(self, reference: str, text: str) -> str:
        """Align text to reference using edit operations."""
        if not reference or not text:
            return text

        # Simple alignment: pad shorter string
        if len(text) < len(reference):
            text = text + " " * (len(reference) - len(text))
        elif len(text) > len(reference):
            text = text[:len(reference)]

        return text


# =============================================================================
# Main Pipeline
# =============================================================================

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
        model_path: str = None
    ):
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.padding = padding

        # Initialize components
        self.ocr_engine = OCREngine(engine, lang, model_path=model_path)
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
        frames = self._load_frames(input_path)
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

    def _load_frames(
        self,
        input_path: str
    ) -> List[Tuple[int, np.ndarray, str]]:
        """Load frames from video or folder."""
        path = Path(input_path)
        frames = []

        if path.is_file():
            # Video file
            if path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                cap = cv2.VideoCapture(str(path))
                frame_idx = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_path = f"video_frame_{frame_idx:04d}"
                    frames.append((frame_idx, frame, frame_path))
                    frame_idx += 1

                cap.release()
            else:
                # Single image
                image = cv2.imread(str(path))
                if image is not None:
                    frames.append((0, image, str(path)))

        elif path.is_dir():
            # Folder of images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            image_files = sorted([
                f for f in path.iterdir()
                if f.suffix.lower() in image_extensions
            ])

            for idx, img_path in enumerate(image_files):
                image = cv2.imread(str(img_path))
                if image is not None:
                    frames.append((idx, image, str(img_path)))

        return frames

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
        annotated = self._draw_annotations(image, detected_lines)
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

    def _draw_annotations(
        self,
        image: np.ndarray,
        lines: List[DetectedLine]
    ) -> np.ndarray:
        """Draw line boxes and labels on image."""
        annotated = image.copy()

        for line in lines:
            bbox = line.bbox

            # Draw rectangle
            cv2.rectangle(
                annotated,
                (bbox.x, bbox.y),
                (bbox.x + bbox.width, bbox.y + bbox.height),
                (0, 255, 0), 2
            )

            # Draw label
            label = f"L{line.index}: {line.text[:30]}..." if len(line.text) > 30 else f"L{line.index}: {line.text}"
            cv2.putText(
                annotated, label,
                (bbox.x, bbox.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1
            )

            # Draw confidence
            conf_label = f"{line.confidence:.2f}"
            cv2.putText(
                annotated, conf_label,
                (bbox.x + bbox.width - 40, bbox.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 0), 1
            )

        return annotated

    def _save_outputs(
        self,
        result: TemporalResult,
        frame_results: List[FrameResult],
        out_path: Path
    ):
        """Save JSON outputs."""
        # Per-frame results
        frame_data = []
        for fr in frame_results:
            frame_data.append({
                "frame_index": fr.frame_index,
                "frame_path": fr.frame_path,
                "quality_score": fr.quality_score,
                "is_valid": fr.is_valid,
                "skew_angle": fr.skew_angle,
                "lines": [
                    {
                        "index": line.index,
                        "bbox": [line.bbox.x, line.bbox.y, line.bbox.width, line.bbox.height],
                        "text": line.text,
                        "confidence": line.confidence,
                        "crop_path": line.crop_path
                    }
                    for line in fr.lines
                ]
            })

        with open(out_path / "frame_results.json", "w") as f:
            json.dump(frame_data, f, indent=2)

        # Aggregated results
        agg_data = {
            "full_text": result.full_text,
            "frames_processed": result.frames_processed,
            "frames_used": result.frames_used,
            "aggregation_method": result.aggregation_method,
            "lines": [
                {
                    "line_index": line.line_index,
                    "final_text": line.final_text,
                    "confidence": line.confidence,
                    "consensus_text": line.consensus_text,
                    "best_single_text": line.best_single_text,
                    "best_single_confidence": line.best_single_confidence,
                    "contributing_frames": line.contributing_frames,
                    "num_candidates": len(line.all_candidates)
                }
                for line in result.lines
            ]
        }

        with open(out_path / "aggregated_results.json", "w") as f:
            json.dump(agg_data, f, indent=2)

        print(f"\n[Output] Saved results to {out_path}")


# =============================================================================
# Single-Image Mode (Line-Crop OCR)
# =============================================================================

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
        vision_mode: str = "verify_all"
    ):
        self.padding = padding
        self.deskew = deskew

        self.ocr_engine = OCREngine(engine, lang, model_path=model_path)
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
        json_path = out_path / f"{image_name}_result.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

        return result


# =============================================================================
# CLI Interface
# =============================================================================

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
        "--engine", "-e", choices=["paddle", "easyocr", "tesseract", "trocr"], default="paddle",
        help="OCR engine (default: paddle). Use 'trocr' for handwriting."
    )
    parser.add_argument(
        "--model_path", default=None,
        help="Path to custom TrOCR model (only used with --engine trocr)"
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

    print(f"[Temporal OCR] Mode: {mode}")
    print(f"[Temporal OCR] Input: {args.input}")
    print(f"[Temporal OCR] Output: {args.out_dir}")
    print(f"[Temporal OCR] Engine: {args.engine}")
    print()

    use_consensus = args.use_consensus.lower() in ["true", "yes", "1"]

    if mode == "temporal":
        # Temporal aggregation mode
        pipeline = TemporalOCRPipeline(
            engine=args.engine,
            max_frames=args.max_frames,
            min_frames=args.min_frames,
            use_consensus=use_consensus,
            blur_threshold=args.blur_threshold,
            padding=args.padding,
            lang=args.lang,
            adaptive_thresh=args.adaptive_thresh,
            model_path=args.model_path
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
            engine=args.engine,
            padding=args.padding,
            lang=args.lang,
            adaptive_thresh=args.adaptive_thresh,
            deskew=not args.no_deskew,
            model_path=args.model_path,
            vision_verify=args.vision_verify,
            vision_model=args.vision_model,
            vision_mode=args.vision_mode
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
                print(f"Skew angle: {result['skew_angle']:.1f}°")
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
