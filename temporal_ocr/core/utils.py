"""
Utility functions and data classes for Temporal OCR.

Contains shared data structures, file I/O helpers, and configuration utilities.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import cv2
import numpy as np


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
# File I/O Utilities
# =============================================================================

def load_frames(input_path: str) -> List[Tuple[int, np.ndarray, str]]:
    """
    Load frames from video or folder.

    Args:
        input_path: Path to video file or folder of images

    Returns:
        List of (frame_index, image, frame_path)
    """
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


def save_frame_results(frame_results: List[FrameResult], out_path: Path) -> None:
    """Save per-frame results to JSON."""
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


def save_aggregated_results(result: TemporalResult, out_path: Path) -> None:
    """Save aggregated results to JSON."""
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


def save_single_image_result(result: Dict[str, Any], out_path: Path, image_name: str) -> None:
    """Save single image OCR result to JSON."""
    json_path = out_path / f"{image_name}_result.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)


def draw_annotations(
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
