#!/usr/bin/env python3
"""
Advanced Data Augmentation for Handwriting OCR

Implements augmentation techniques specifically designed for handwriting:
- Elastic distortion (simulates natural handwriting variation)
- Gaussian blur (simulates camera focus issues)
- Salt-and-pepper noise (simulates paper texture/artifacts)
- Perspective transforms (simulates camera angle)
- Color jitter (simulates different ink/paper colors)
- Morphological operations (thin/thicken strokes)
"""

import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from typing import Tuple, Optional
import cv2


class HandwritingAugmentor:
    """
    Augmentation pipeline optimized for handwriting OCR.

    Usage:
        augmentor = HandwritingAugmentor(strength='medium')
        augmented_img = augmentor(original_img)
    """

    def __init__(
        self,
        strength: str = 'medium',
        always_apply: bool = False
    ):
        """
        Initialize augmentor.

        Args:
            strength: 'light', 'medium', or 'heavy'
            always_apply: If True, always apply augmentations
        """
        self.strength = strength
        self.always_apply = always_apply

        # Set parameters based on strength
        if strength == 'light':
            self.aug_probability = 0.3
            self.rotation_range = (-3, 3)
            self.blur_range = (0, 1)
            self.noise_amount = 0.01
            self.elastic_alpha = 20
            self.elastic_sigma = 3
        elif strength == 'medium':
            self.aug_probability = 0.5
            self.rotation_range = (-5, 5)
            self.blur_range = (0, 2)
            self.noise_amount = 0.02
            self.elastic_alpha = 40
            self.elastic_sigma = 4
        else:  # heavy
            self.aug_probability = 0.7
            self.rotation_range = (-10, 10)
            self.blur_range = (0, 3)
            self.noise_amount = 0.03
            self.elastic_alpha = 60
            self.elastic_sigma = 5

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply random augmentations to image."""
        # Convert to numpy for some operations
        img_array = np.array(image)

        # Apply augmentations probabilistically
        if self._should_apply():
            img_array = self._elastic_distortion(img_array)

        if self._should_apply():
            image = self._gaussian_blur(Image.fromarray(img_array))
            img_array = np.array(image)

        if self._should_apply():
            img_array = self._add_noise(img_array)

        if self._should_apply():
            image = self._color_jitter(Image.fromarray(img_array))
            img_array = np.array(image)

        if self._should_apply():
            image = self._random_rotation(Image.fromarray(img_array))
            img_array = np.array(image)

        if self._should_apply():
            img_array = self._morphological_transform(img_array)

        if self._should_apply():
            img_array = self._perspective_transform(img_array)

        return Image.fromarray(img_array)

    def _should_apply(self) -> bool:
        """Determine if augmentation should be applied."""
        if self.always_apply:
            return True
        return random.random() < self.aug_probability

    def _elastic_distortion(self, image: np.ndarray) -> np.ndarray:
        """
        Apply elastic distortion to simulate natural handwriting variation.

        This is one of the most effective augmentations for handwriting.
        """
        if len(image.shape) == 2:
            h, w = image.shape
            channels = 1
        else:
            h, w, channels = image.shape

        # Generate random displacement fields
        dx = np.random.randn(h, w) * self.elastic_alpha
        dy = np.random.randn(h, w) * self.elastic_alpha

        # Smooth the displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), self.elastic_sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), self.elastic_sigma)

        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Apply displacement
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        # Remap image
        if channels == 1:
            return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        else:
            return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    def _gaussian_blur(self, image: Image.Image) -> Image.Image:
        """Apply Gaussian blur to simulate focus issues."""
        radius = random.uniform(*self.blur_range)
        if radius > 0:
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image

    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add salt-and-pepper noise to simulate paper texture."""
        noisy = image.copy()

        # Salt (white pixels)
        salt_mask = np.random.random(image.shape[:2]) < self.noise_amount / 2
        if len(image.shape) == 3:
            salt_mask = np.stack([salt_mask] * image.shape[2], axis=-1)
        noisy[salt_mask] = 255

        # Pepper (black pixels)
        pepper_mask = np.random.random(image.shape[:2]) < self.noise_amount / 2
        if len(image.shape) == 3:
            pepper_mask = np.stack([pepper_mask] * image.shape[2], axis=-1)
        noisy[pepper_mask] = 0

        return noisy

    def _color_jitter(self, image: Image.Image) -> Image.Image:
        """Apply color jittering (brightness, contrast, saturation)."""
        # Brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))

        # Contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))

        # Sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))

        return image

    def _random_rotation(self, image: Image.Image) -> Image.Image:
        """Apply small random rotation."""
        angle = random.uniform(*self.rotation_range)
        return image.rotate(angle, fillcolor='white', resample=Image.BILINEAR)

    def _morphological_transform(self, image: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to thin/thicken strokes.

        Simulates different pen pressures and writing styles.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Random kernel size
        kernel_size = random.choice([1, 2, 3])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Randomly dilate or erode
        if random.random() > 0.5:
            # Dilate (thicken strokes)
            result = cv2.dilate(gray, kernel, iterations=1)
        else:
            # Erode (thin strokes)
            result = cv2.erode(gray, kernel, iterations=1)

        # Convert back to RGB if needed
        if len(image.shape) == 3:
            return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        return result

    def _perspective_transform(self, image: np.ndarray) -> np.ndarray:
        """Apply small perspective transform to simulate camera angle."""
        h, w = image.shape[:2]

        # Define small random perspective shift
        margin = 0.05  # 5% of image size
        shift = int(min(w, h) * margin)

        # Source points (original corners)
        src = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])

        # Destination points (slightly shifted)
        dst = np.float32([
            [random.randint(0, shift), random.randint(0, shift)],
            [w - random.randint(0, shift), random.randint(0, shift)],
            [w - random.randint(0, shift), h - random.randint(0, shift)],
            [random.randint(0, shift), h - random.randint(0, shift)]
        ])

        # Compute and apply transform
        M = cv2.getPerspectiveTransform(src, dst)

        if len(image.shape) == 3:
            return cv2.warpPerspective(image, M, (w, h),
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(255, 255, 255))
        else:
            return cv2.warpPerspective(image, M, (w, h),
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=255)


def deskew_image(image: np.ndarray, max_angle: float = 10) -> np.ndarray:
    """
    Deskew image by detecting and correcting text angle.

    Uses Hough line detection to find dominant angle.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find edges
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None:
        return image

    # Calculate dominant angle
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90
        if abs(angle) < max_angle:
            angles.append(angle)

    if not angles:
        return image

    # Use median angle
    median_angle = np.median(angles)

    # Rotate to correct
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

    if len(image.shape) == 3:
        return cv2.warpAffine(image, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
    else:
        return cv2.warpAffine(image, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=255)


def denoise_image(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """Apply denoising to clean up image artifacts."""
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    else:
        return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)


def binarize_adaptive(image: np.ndarray, block_size: int = 11) -> np.ndarray:
    """
    Adaptive binarization for varying lighting conditions.

    Uses Sauvola method which works well for handwriting.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Adaptive threshold (Gaussian weighted)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, 2
    )

    if len(image.shape) == 3:
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    return binary


def preprocess_for_ocr(
    image: np.ndarray,
    deskew: bool = True,
    denoise: bool = True,
    binarize: bool = False
) -> np.ndarray:
    """
    Full preprocessing pipeline for OCR.

    Args:
        image: Input image as numpy array
        deskew: Whether to correct rotation
        denoise: Whether to remove noise
        binarize: Whether to convert to binary (not always needed for TrOCR)

    Returns:
        Preprocessed image
    """
    result = image.copy()

    if deskew:
        result = deskew_image(result)

    if denoise:
        result = denoise_image(result)

    if binarize:
        result = binarize_adaptive(result)

    return result


if __name__ == "__main__":
    # Test augmentation
    import sys

    if len(sys.argv) < 2:
        print("Usage: python augmentation.py <image_path>")
        sys.exit(1)

    img = Image.open(sys.argv[1]).convert('RGB')

    # Test augmentor
    augmentor = HandwritingAugmentor(strength='medium')

    # Generate 5 augmented versions
    for i in range(5):
        aug_img = augmentor(img)
        aug_img.save(f"augmented_{i}.jpg")
        print(f"Saved augmented_{i}.jpg")

    # Test preprocessing
    img_array = np.array(img)
    preprocessed = preprocess_for_ocr(img_array)
    Image.fromarray(preprocessed).save("preprocessed.jpg")
    print("Saved preprocessed.jpg")
