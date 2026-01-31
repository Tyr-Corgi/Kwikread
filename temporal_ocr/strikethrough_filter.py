"""
Strikethrough Detection Module for Temporal OCR

Detects and filters crossed-out/struck-through text in handwritten images.
Uses horizontal line detection to identify strikethrough marks.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class StrikethroughRegion:
    """A detected strikethrough region."""
    x_start: int
    x_end: int
    y_center: int
    confidence: float


class StrikethroughDetector:
    """
    Detects strikethrough marks in text images.

    Uses morphological operations and line detection to find
    horizontal lines that cross through text.
    """

    def __init__(
        self,
        min_line_length_ratio: float = 0.10,  # Min line length as ratio of image width
        line_thickness_range: Tuple[int, int] = (1, 20),  # Expected line thickness in pixels
        vertical_position_range: Tuple[float, float] = (0.30, 0.70),  # Where strikethrough appears (ratio of height)
    ):
        self.min_line_length_ratio = min_line_length_ratio
        self.line_thickness_range = line_thickness_range
        self.vertical_position_range = vertical_position_range

    def detect_strikethrough(self, image: np.ndarray) -> List[StrikethroughRegion]:
        """
        Detect strikethrough lines in an image.

        Args:
            image: BGR or grayscale image of a text line

        Returns:
            List of detected strikethrough regions
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape

        # Threshold to binary (inverted - text is white)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Method 1: Morphological line detection
        regions = self._detect_via_morphology(binary, w, h)

        # Method 2: Hough line detection (backup)
        if not regions:
            regions = self._detect_via_hough(binary, w, h)

        return regions

    def _detect_via_morphology(
        self,
        binary: np.ndarray,
        width: int,
        height: int
    ) -> List[StrikethroughRegion]:
        """Detect horizontal lines using morphological operations."""
        regions = []

        # Create horizontal kernel to detect horizontal lines
        # Use smaller kernel for better detection of strikethrough
        min_line_length = int(width * self.min_line_length_ratio)
        kernel_length = max(20, min_line_length // 3)  # Smaller kernel for sensitivity
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        # Find contours of detected lines
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w_box, h_box = cv2.boundingRect(contour)

            # Check if it's a horizontal line (width >> height)
            if w_box < min_line_length:
                continue
            if h_box > self.line_thickness_range[1]:
                continue

            # Check vertical position (strikethrough usually in middle of text)
            y_center_ratio = (y + h_box / 2) / height
            if not (self.vertical_position_range[0] <= y_center_ratio <= self.vertical_position_range[1]):
                continue

            # Calculate confidence based on line characteristics
            # Higher confidence for lines that are clearly horizontal (width >> height)
            aspect_ratio = w_box / max(h_box, 1)
            aspect_score = min(1.0, aspect_ratio / 20)  # Good strikethrough has aspect ratio > 20
            length_score = min(1.0, w_box / (width * 0.3))
            position_score = 1.0 - abs(y_center_ratio - 0.5) * 2
            confidence = (aspect_score * 0.4 + length_score * 0.3 + position_score * 0.3)

            regions.append(StrikethroughRegion(
                x_start=x,
                x_end=x + w_box,
                y_center=y + h_box // 2,
                confidence=confidence
            ))

        return regions

    def _detect_via_hough(
        self,
        binary: np.ndarray,
        width: int,
        height: int
    ) -> List[StrikethroughRegion]:
        """Detect horizontal lines using Hough transform."""
        regions = []

        # Edge detection
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)

        # Hough line detection
        min_line_length = int(width * self.min_line_length_ratio)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=min_line_length,
            maxLineGap=10
        )

        if lines is None:
            return regions

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Check if line is approximately horizontal (within 10 degrees)
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle > 10 and angle < 170:
                continue

            # Check vertical position
            y_center = (y1 + y2) / 2
            y_center_ratio = y_center / height
            if not (self.vertical_position_range[0] <= y_center_ratio <= self.vertical_position_range[1]):
                continue

            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            confidence = min(1.0, line_length / (width * 0.5))

            regions.append(StrikethroughRegion(
                x_start=min(x1, x2),
                x_end=max(x1, x2),
                y_center=int(y_center),
                confidence=confidence
            ))

        return regions

    def has_strikethrough(self, image: np.ndarray, min_confidence: float = 0.5) -> bool:
        """Check if image contains strikethrough text."""
        regions = self.detect_strikethrough(image)
        return any(r.confidence >= min_confidence for r in regions)

    def get_strikethrough_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Get a mask indicating strikethrough regions.

        Returns:
            Binary mask where 255 = strikethrough region, 0 = normal text
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        regions = self.detect_strikethrough(image)
        for region in regions:
            if region.confidence >= 0.4:
                # Mark the region around the strikethrough line
                y_top = max(0, region.y_center - h // 4)
                y_bottom = min(h, region.y_center + h // 4)
                mask[y_top:y_bottom, region.x_start:region.x_end] = 255

        return mask


def filter_strikethrough_text(
    image: np.ndarray,
    text: str,
    detector: Optional[StrikethroughDetector] = None
) -> Tuple[str, bool, float]:
    """
    Filter out struck-through portions of text.

    This function analyzes the image to find strikethrough regions,
    then attempts to identify which part of the recognized text
    corresponds to the struck-through portion.

    Args:
        image: The line crop image
        text: The OCR-recognized text
        detector: Optional StrikethroughDetector instance

    Returns:
        Tuple of (filtered_text, had_strikethrough, strikethrough_ratio)
    """
    if detector is None:
        detector = StrikethroughDetector()

    regions = detector.detect_strikethrough(image)

    if not regions:
        return text, False, 0.0

    # Calculate what portion of the image is struck through
    h, w = image.shape[:2]
    total_strikethrough_width = sum(r.x_end - r.x_start for r in regions if r.confidence >= 0.4)
    strikethrough_ratio = total_strikethrough_width / w

    # If most of the line is struck through, return empty
    if strikethrough_ratio > 0.8:
        return "", True, strikethrough_ratio

    # Find the main strikethrough region
    main_region = max(regions, key=lambda r: (r.x_end - r.x_start) * r.confidence)

    if main_region.confidence < 0.4:
        return text, False, 0.0

    # Estimate which portion of text is struck through based on position
    strike_start_ratio = main_region.x_start / w
    strike_end_ratio = main_region.x_end / w

    # If strikethrough is at the beginning (first 40% of image)
    if strike_end_ratio < 0.5:
        # Remove first word(s) that fall within the struck region
        words = text.split()
        if len(words) > 1:
            # Estimate how many words to remove based on strikethrough coverage
            words_to_remove = max(1, int(len(words) * strike_end_ratio * 1.2))
            filtered_text = " ".join(words[words_to_remove:])
            return filtered_text, True, strikethrough_ratio

    # If strikethrough is at the end
    elif strike_start_ratio > 0.5:
        words = text.split()
        if len(words) > 1:
            words_to_keep = max(1, int(len(words) * strike_start_ratio * 0.8))
            filtered_text = " ".join(words[:words_to_keep])
            return filtered_text, True, strikethrough_ratio

    # Strikethrough in middle - harder to handle, just flag it
    return text, True, strikethrough_ratio


def detect_and_report_strikethrough(image: np.ndarray) -> dict:
    """
    Analyze an image for strikethrough and return detailed report.

    Args:
        image: The line crop image

    Returns:
        Dictionary with detection results
    """
    detector = StrikethroughDetector()
    regions = detector.detect_strikethrough(image)

    h, w = image.shape[:2]

    return {
        "has_strikethrough": len(regions) > 0,
        "num_regions": len(regions),
        "regions": [
            {
                "x_start": r.x_start,
                "x_end": r.x_end,
                "y_center": r.y_center,
                "width": r.x_end - r.x_start,
                "confidence": r.confidence,
                "coverage_ratio": (r.x_end - r.x_start) / w
            }
            for r in regions
        ],
        "total_coverage": sum(r.x_end - r.x_start for r in regions) / w if regions else 0.0
    }
