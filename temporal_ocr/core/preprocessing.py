"""
Image preprocessing functions for Temporal OCR.

Contains image enhancement, deskewing, and normalization utilities
optimized for handwriting OCR.
"""

import cv2
import numpy as np


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
