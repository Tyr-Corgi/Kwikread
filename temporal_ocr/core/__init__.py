"""
Core module for Temporal OCR.

This package contains modular components for OCR processing:
- utils: Data classes, file I/O, and configuration
- preprocessing: Image enhancement and deskewing
- detection: Text detection and line grouping
- recognition: OCR engine wrappers (PaddleOCR, EasyOCR, TrOCR, etc.)
- postprocessing: Quality assessment, temporal matching, consensus voting
- video: Main pipeline classes for temporal and single-image processing
"""

# Data classes
from .utils import (
    BoundingBox,
    DetectedLine,
    FrameResult,
    AggregatedLine,
    TemporalResult,
)

# File I/O utilities
from .utils import (
    load_frames,
    save_frame_results,
    save_aggregated_results,
    save_single_image_result,
    draw_annotations,
)

# Preprocessing
from .preprocessing import ImagePreprocessor

# Detection
from .detection import LineDetector, split_wide_detections

# Recognition
from .recognition import OCREngine, EnsembleOCREngine

# Postprocessing
from .postprocessing import (
    FrameQualityAssessor,
    TemporalLineMatcher,
    TemporalAggregator,
)

# Main pipelines
from .video import TemporalOCRPipeline, LineCropOCRPipeline


__all__ = [
    # Data classes
    "BoundingBox",
    "DetectedLine",
    "FrameResult",
    "AggregatedLine",
    "TemporalResult",
    # File I/O
    "load_frames",
    "save_frame_results",
    "save_aggregated_results",
    "save_single_image_result",
    "draw_annotations",
    # Preprocessing
    "ImagePreprocessor",
    # Detection
    "LineDetector",
    "split_wide_detections",
    # Recognition
    "OCREngine",
    "EnsembleOCREngine",
    # Postprocessing
    "FrameQualityAssessor",
    "TemporalLineMatcher",
    "TemporalAggregator",
    # Pipelines
    "TemporalOCRPipeline",
    "LineCropOCRPipeline",
]
