"""
Fast OCR Server - Keeps TrOCR model loaded for instant inference.

Usage:
  # Start server (in background)
  python ocr_server.py serve &

  # Process image (uses running server)
  python ocr_server.py process image.jpg

  # Process with selective vision verification (1-2 lowest confidence lines)
  python ocr_server.py process image.jpg --verify

  # Or via HTTP
  curl -X POST http://localhost:8765/ocr -F "image=@image.jpg"
"""

import sys
import json
import time
import socket
import struct
import base64
import os
import traceback
import threading
import requests
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

# CPU optimization constants
NUM_WORKERS = 4   # Parallel workers for preprocessing
BATCH_SIZE = 32   # Larger batches are faster on CPU

# Server config
HOST = 'localhost'
PORT = 8765
SOCKET_PATH = '/tmp/ocr_server.sock'
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434/api/generate')

# Security: Input validation limits
MAX_PATH_LENGTH = 10 * 1024  # 10KB max for paths
MAX_REQUEST_SIZE = 64 * 1024  # 64KB max for entire request
MAX_ACTION_LENGTH = 32  # Max length for action field

# Security: Rate limiting (token bucket)
RATE_LIMIT_REQUESTS = 100  # Max requests per window
RATE_LIMIT_WINDOW = 60  # Window size in seconds

# =============================================================================
# Line Detection Constants
# =============================================================================
# DBSCAN clustering parameters for grouping text boxes into lines
MIN_DBSCAN_EPS = 20       # Minimum epsilon for DBSCAN clustering (pixels)
MAX_DBSCAN_EPS = 35       # Maximum epsilon for DBSCAN clustering (pixels)
DBSCAN_EPS_RATIO = 0.3    # Ratio of average text height to use as epsilon

# Column detection parameters
COLUMN_GAP_RATIO = 0.15   # Ratio of image width that indicates a column gap

# Minimum dimensions for valid text boxes
MIN_BOX_WIDTH = 50        # Minimum width for deskew processing (pixels)
MIN_BOX_HEIGHT = 15       # Minimum height for deskew processing (pixels)

# Security: Allowed directories for file access (prevents path traversal attacks)
# Add more directories as needed, or set to None to allow all (not recommended)
ALLOWED_DIRECTORIES = [
    Path(__file__).parent.resolve(),  # temporal_ocr directory
    Path(__file__).parent.parent.resolve(),  # kwikread directory
    Path("/tmp"),  # Temp files
]
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

# Model paths - fine-tuned model preferred if available (v4 trained on 6800+ IAM samples with augmentation)
FINETUNED_MODEL_PATH = Path(__file__).parent / "finetune" / "model_v4" / "final"
FINETUNED_MODEL_V3_PATH = Path(__file__).parent / "finetune" / "model_v3" / "final"
FINETUNED_MODEL_V2_PATH = Path(__file__).parent / "finetune" / "model_v2" / "final"
FINETUNED_MODEL_V1_PATH = Path(__file__).parent / "finetune" / "model" / "final"
BASE_MODEL_NAME = 'microsoft/trocr-base-handwritten'


class PathSecurityError(Exception):
    """Raised when a file path fails security validation."""
    pass


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    pass


class InputValidationError(Exception):
    """Raised when input validation fails."""
    pass


class RateLimiter:
    """Simple sliding window rate limiter for DoS protection.

    Uses a per-client token bucket approach with automatic cleanup
    of stale entries to prevent memory exhaustion.
    """

    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window_seconds: int = RATE_LIMIT_WINDOW):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = {}  # client_id -> list of timestamps
        self._lock = threading.Lock()

    def _cleanup_old_entries(self, now: float):
        """Remove entries older than the window. Called with lock held."""
        cutoff = now - self.window_seconds
        stale_clients = []
        for client_id, timestamps in self._requests.items():
            # Filter out old timestamps
            self._requests[client_id] = [t for t in timestamps if t > cutoff]
            if not self._requests[client_id]:
                stale_clients.append(client_id)
        # Remove clients with no recent requests
        for client_id in stale_clients:
            del self._requests[client_id]

    def check_rate_limit(self, client_id: str = "default") -> bool:
        """Check if request is allowed under rate limit.

        Args:
            client_id: Identifier for the client (default for Unix sockets)

        Returns:
            True if request is allowed

        Raises:
            RateLimitError: If rate limit is exceeded
        """
        now = time.time()

        with self._lock:
            # Periodic cleanup (every 10 requests)
            if sum(len(v) for v in self._requests.values()) % 10 == 0:
                self._cleanup_old_entries(now)

            cutoff = now - self.window_seconds

            if client_id not in self._requests:
                self._requests[client_id] = []

            # Filter timestamps within window
            recent = [t for t in self._requests[client_id] if t > cutoff]

            if len(recent) >= self.max_requests:
                raise RateLimitError(
                    f"Rate limit exceeded: {self.max_requests} requests per {self.window_seconds}s"
                )

            # Record this request
            recent.append(now)
            self._requests[client_id] = recent

        return True


def validate_request_input(request: dict, data_length: int) -> None:
    """Validate request input for security.

    Args:
        request: Parsed JSON request dict
        data_length: Length of raw request data in bytes

    Raises:
        InputValidationError: If validation fails
    """
    # Check overall request size
    if data_length > MAX_REQUEST_SIZE:
        raise InputValidationError(
            f"Request too large: {data_length} bytes (max: {MAX_REQUEST_SIZE})"
        )

    # Validate action field
    action = request.get('action', '')
    if not isinstance(action, str):
        raise InputValidationError("Action must be a string")
    if len(action) > MAX_ACTION_LENGTH:
        raise InputValidationError(
            f"Action too long: {len(action)} chars (max: {MAX_ACTION_LENGTH})"
        )

    # Validate image_path if present
    image_path = request.get('image_path', '')
    if image_path:
        if not isinstance(image_path, str):
            raise InputValidationError("image_path must be a string")
        if len(image_path) > MAX_PATH_LENGTH:
            raise InputValidationError(
                f"Path too long: {len(image_path)} chars (max: {MAX_PATH_LENGTH})"
            )

    # Validate numeric fields
    max_verify = request.get('max_verify')
    if max_verify is not None:
        if not isinstance(max_verify, (int, float)) or max_verify < 0 or max_verify > 100:
            raise InputValidationError("max_verify must be a number between 0 and 100")


def validate_image_path(image_path: str) -> Path:
    """
    Validate that an image path is safe to access.

    Security checks:
    1. Path must resolve to an absolute path (no ../ traversal)
    2. Path must be within allowed directories
    3. File extension must be an allowed image type
    4. Path must not contain null bytes or other injection attempts

    Args:
        image_path: The path to validate

    Returns:
        Resolved absolute Path object

    Raises:
        PathSecurityError: If validation fails
    """
    # Check for null bytes (injection attempt)
    if '\x00' in image_path:
        raise PathSecurityError("Invalid path: contains null bytes")

    # Convert to Path and resolve to absolute (eliminates ../)
    try:
        path = Path(image_path).resolve()
    except (OSError, ValueError) as e:
        raise PathSecurityError(f"Invalid path format: {e}")

    # Check file extension
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise PathSecurityError(
            f"Invalid file type: {path.suffix}. "
            f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Check if path is within allowed directories
    if ALLOWED_DIRECTORIES is not None:
        path_allowed = False
        for allowed_dir in ALLOWED_DIRECTORIES:
            try:
                path.relative_to(allowed_dir)
                path_allowed = True
                break
            except ValueError:
                continue

        if not path_allowed:
            raise PathSecurityError(
                f"Access denied: path outside allowed directories. "
                f"Path: {path}"
            )

    return path


def preprocess_crop(crop_img: np.ndarray, style: str = 'auto') -> Image.Image:
    """
    Preprocess a line crop for better TrOCR recognition.

    Args:
        crop_img: BGR numpy array of the cropped line
        style: Preprocessing style - 'auto', 'light', 'heavy', or 'cursive'

    Returns:
        Preprocessed PIL Image ready for TrOCR
    """
    # Convert to grayscale for analysis
    if len(crop_img.shape) == 3:
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_img.copy()

    h, w = gray.shape

    # Auto-detect preprocessing needs based on image characteristics
    if style == 'auto':
        # Calculate contrast and brightness metrics
        mean_val = np.mean(gray)
        std_val = np.std(gray)

        # Detect if image has low contrast (light handwriting)
        low_contrast = std_val < 40

        # Detect if text is very light (high mean = mostly white)
        light_text = mean_val > 200

        # Detect connected/cursive text by checking horizontal continuity
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 4, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        cursive_ratio = np.sum(horizontal_lines > 0) / max(1, np.sum(binary > 0))
        is_cursive = cursive_ratio > 0.15

        # Choose style based on detection
        if low_contrast or light_text:
            style = 'heavy'
        elif is_cursive:
            style = 'cursive'
        else:
            style = 'light'

    # Apply preprocessing based on style
    processed = crop_img.copy()

    if style in ('heavy', 'cursive'):
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This enhances local contrast without over-amplifying noise
        if len(processed.shape) == 3:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
        else:
            l_channel = processed

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)

        if len(processed.shape) == 3:
            lab[:, :, 0] = enhanced_l
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            processed = enhanced_l

    # Deskew if needed (for tilted handwriting)
    processed = deskew_crop(processed)

    if style == 'cursive':
        # For cursive text, apply mild denoising that preserves stroke details
        if len(processed.shape) == 3:
            gray_proc = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray_proc = processed

        # Gentle denoising that preserves edges
        denoised = cv2.fastNlMeansDenoising(gray_proc, None, h=5, templateWindowSize=7, searchWindowSize=21)

        # Convert back to 3-channel for consistency
        processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    elif style == 'heavy':
        # For low contrast, apply stronger enhancement
        if len(processed.shape) == 3:
            gray_proc = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray_proc = processed

        # Bilateral filter preserves edges while smoothing
        filtered = cv2.bilateralFilter(gray_proc, 9, 75, 75)

        # Adaptive thresholding for very light text
        mean_val = np.mean(filtered)
        if mean_val > 210:  # Very light text
            # Use adaptive threshold to enhance faint strokes
            adaptive = cv2.adaptiveThreshold(
                filtered, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=15,
                C=8
            )
            # Blend with original to preserve some grayscale info
            blended = cv2.addWeighted(filtered, 0.3, adaptive, 0.7, 0)
            processed = cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)
        else:
            processed = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

    # Convert to PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

    # Final PIL-based enhancements
    if style in ('heavy', 'cursive'):
        # Slight sharpening to make strokes clearer
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.2)

    return pil_img


def deskew_crop(image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """
    Deskew a line crop if it's tilted.

    Args:
        image: BGR or grayscale numpy array
        max_angle: Maximum angle to correct (degrees)

    Returns:
        Deskewed image
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    # Skip small images
    if w < MIN_BOX_WIDTH or h < MIN_BOX_HEIGHT:
        return image

    # Threshold to binary
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find coordinates of text pixels
    coords = np.column_stack(np.where(binary > 0))

    if len(coords) < 10:
        return image

    # Fit a minimum area rectangle
    try:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # Adjust angle interpretation
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        # Only correct if angle is significant but not too extreme
        if abs(angle) < 0.5 or abs(angle) > max_angle:
            return image

        # Rotate image to correct skew
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding box size
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Adjust rotation matrix for new size
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2

        # Apply rotation with white background
        rotated = cv2.warpAffine(
            image, rotation_matrix, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) if len(image.shape) == 3 else 255
        )

        return rotated

    except (cv2.error, ValueError, TypeError) as e:
        # Return original image if deskew fails (e.g., invalid contours)
        print(f"[Deskew] Could not deskew image: {e}")
        return image


def calculate_adaptive_padding(line_height: int, image_width: int) -> Tuple[int, int]:
    """
    Calculate adaptive padding based on text characteristics.

    Args:
        line_height: Height of the detected line in pixels
        image_width: Width of the full image

    Returns:
        Tuple of (horizontal_padding, vertical_padding) in pixels
    """
    # Base vertical padding scales with text size
    base_padding_y = max(8, int(line_height * 0.25))

    # Horizontal padding should be more generous to capture word edges
    base_padding_x = max(15, int(line_height * 0.5))

    # Additional padding for potentially connected text
    # Larger text often has more spacing needs
    if line_height > 40:
        base_padding_x = int(base_padding_x * 1.2)
        base_padding_y = int(base_padding_y * 1.1)

    # Cap padding at reasonable maximum
    max_padding_x = min(40, image_width // 15)
    max_padding_y = min(25, image_width // 30)

    return (min(base_padding_x, max_padding_x), min(base_padding_y, max_padding_y))


class OCRServer:
    """Persistent OCR server with pre-loaded models."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.detector = None  # EasyOCR for line detection
        self._loaded = False

    def load_models(self):
        """Load all models once at startup."""
        if self._loaded:
            return

        print("[Server] Loading models...")
        t0 = time.time()

        # TrOCR for recognition
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch

        # Optimize torch for CPU inference
        torch.set_num_threads(NUM_WORKERS)
        torch.set_num_interop_threads(2)

        # Use fine-tuned model if available, otherwise fall back to base model
        # Prefer v2 (IAM + grocery) > v1 (grocery only) > base model
        if FINETUNED_MODEL_PATH.exists():
            model_path = str(FINETUNED_MODEL_PATH)
            print(f"[Server] Using fine-tuned model v2 (IAM + grocery): {model_path}")
        elif FINETUNED_MODEL_V1_PATH.exists():
            model_path = str(FINETUNED_MODEL_V1_PATH)
            print(f"[Server] Using fine-tuned model v1 (grocery): {model_path}")
        else:
            model_path = BASE_MODEL_NAME
            print(f"[Server] Using base model: {model_path}")

        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)

        # Use MPS (Metal) on Mac, CUDA on Linux/Windows, CPU fallback
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model.to(self.device)
        self.model.eval()

        # Initialize thread pool for parallel preprocessing
        self.executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)

        # EasyOCR for detection (text boxes only)
        import easyocr
        self.detector = easyocr.Reader(['en'], gpu=self.device != 'cpu')

        # Strikethrough detector
        try:
            from strikethrough_filter import StrikethroughDetector, filter_strikethrough_text
            self.strike_detector = StrikethroughDetector()
            self.filter_strike = filter_strikethrough_text
        except (ImportError, ModuleNotFoundError) as e:
            print(f"[Server] Strikethrough filter not available: {e}")
            self.strike_detector = None
            self.filter_strike = None

        # Grocery corrector - force reload to get latest changes
        try:
            import importlib
            import grocery_corrector
            importlib.reload(grocery_corrector)
            from grocery_corrector import correct_grocery_text, PHRASE_COMPLETIONS
            self.correct_grocery = correct_grocery_text
            print(f"[Server] Grocery corrector loaded with {len(PHRASE_COMPLETIONS)} phrase completions")
            # Verify it works
            test_result = correct_grocery_text("chai", threshold=0.88)
            print(f"[Server] Verification: 'chai' -> {test_result}")
        except Exception as e:
            print(f"[Server] Warning: Could not load grocery_corrector: {e}")
            import traceback
            traceback.print_exc()
            self.correct_grocery = None

        t1 = time.time()
        print(f"[Server] Models loaded in {t1-t0:.1f}s (device: {self.device})")
        self._loaded = True

    def detect_lines(self, image: np.ndarray) -> List[Tuple[list, str, float]]:
        """Detect text lines using EasyOCR with optimized settings."""
        # Use paragraph=False to get individual word boxes for better grouping control
        # Use width_ths=0.7 to be more aggressive about merging horizontal boxes
        # Use height_ths=0.5 to group boxes with similar heights
        results = self.detector.readtext(
            image,
            paragraph=False,      # Get word-level boxes for precise grouping
            width_ths=0.7,        # Higher threshold = more aggressive horizontal merging
            height_ths=0.5,       # Merge boxes of similar height
            ycenter_ths=0.5,      # Y-center threshold for same-line detection
            x_ths=1.0,            # Allow larger horizontal gaps before splitting
            text_threshold=0.5,   # Lower threshold to catch faint text (like "Candles")
            low_text=0.3,         # Lower bound for text region detection
        )
        return results

    def recognize_batch(self, images: List[Image.Image]) -> List[str]:
        """Batch recognize text from PIL images with optimized chunked processing."""
        import torch

        if not images:
            return []

        all_texts = []

        # Process in optimal batch sizes - inference_mode is faster than no_grad
        with torch.inference_mode():
            for i in range(0, len(images), BATCH_SIZE):
                batch = images[i:i + BATCH_SIZE]

                pixel_values = self.processor(
                    images=batch,
                    return_tensors='pt',
                    padding=True
                ).pixel_values.to(self.device)

                # Greedy decoding for speed (<6s requirement)
                # Post-processing corrections handle most OCR errors
                generated_ids = self.model.generate(
                    pixel_values,
                    max_new_tokens=20,  # Grocery items are short (1-3 words)
                    use_cache=True  # Reuse attention states, 1.8x faster
                )

                texts = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                all_texts.extend(texts)

        return all_texts

    def preprocess_crop_parallel(self, crop_data: Tuple) -> Tuple[int, Image.Image]:
        """Preprocess a single crop - used for parallel processing."""
        idx, crop, style = crop_data
        pil_crop = preprocess_crop(crop, style=style)
        return (idx, pil_crop)

    def verify_with_vision(self, crop_img: np.ndarray, trocr_text: str) -> str:
        """Verify/correct text using LLaVA vision model."""
        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', crop_img)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        prompt = f"""Look at this handwritten text image. The OCR read it as: "{trocr_text}"
Is this correct? If not, what does it actually say?
Reply with ONLY the corrected text, nothing else. If correct, reply with the same text."""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    'model': 'llava:7b',
                    'prompt': prompt,
                    'images': [img_b64],
                    'stream': False,
                    'options': {'temperature': 0.1}
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                # Clean up common LLaVA response patterns
                if result.lower().startswith('the text'):
                    result = result.split(':')[-1].strip().strip('"\'')
                return result if result else trocr_text
        except requests.RequestException as e:
            # Network errors: connection refused, timeout, DNS failure, etc.
            print(f"[Vision] Network error: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            # Response parsing errors
            print(f"[Vision] Response parse error: {e}")

        return trocr_text

    def _detect_text_regions(self, image: np.ndarray) -> Tuple[List, float]:
        """Detect text regions using EasyOCR.

        Args:
            image: BGR numpy array of the image

        Returns:
            Tuple of (detections list, detection time in seconds)
        """
        t1 = time.time()
        detections = self.detect_lines(image)
        t2 = time.time()
        return detections, t2 - t1

    def _cluster_boxes_by_y(self, boxes: List[Tuple]) -> List[int]:
        """Cluster text boxes by y-center using DBSCAN.

        Args:
            boxes: List of (x, y, w, h, cy, text, conf) tuples

        Returns:
            List of cluster labels for each box
        """
        from sklearn.cluster import DBSCAN

        if len(boxes) <= 1:
            return [0] * len(boxes)

        # Use conservative eps: ratio of average text height, capped
        avg_height = np.mean([b[3] for b in boxes])  # b[3] is height
        dynamic_eps = min(MAX_DBSCAN_EPS, max(MIN_DBSCAN_EPS, avg_height * DBSCAN_EPS_RATIO))

        y_centers = np.array([[b[4]] for b in boxes])
        clustering = DBSCAN(eps=dynamic_eps, min_samples=1).fit(y_centers)
        return clustering.labels_.tolist()

    def _split_by_column_gap(
        self, line_boxes: List[Tuple], column_gap_threshold: float
    ) -> List[List[Tuple]]:
        """Split a line of boxes by large x-gaps indicating columns.

        Args:
            line_boxes: List of boxes sorted by x coordinate
            column_gap_threshold: Minimum gap width to indicate a column break

        Returns:
            List of box groups, each representing a separate column item
        """
        if not line_boxes:
            return []

        split_lines = []
        current_group = [line_boxes[0]]

        for i in range(1, len(line_boxes)):
            prev_box = line_boxes[i - 1]
            curr_box = line_boxes[i]
            gap = curr_box[0] - (prev_box[0] + prev_box[2])  # x gap between boxes

            if gap > column_gap_threshold:
                # Large gap - start new line (different column)
                split_lines.append(current_group)
                current_group = [curr_box]
            else:
                # Keep words together even with moderate gaps
                current_group.append(curr_box)

        split_lines.append(current_group)
        return split_lines

    def _calculate_line_bbox(self, group_boxes: List[Tuple]) -> Tuple[int, int, int, int]:
        """Calculate the bounding box for a group of text boxes.

        Args:
            group_boxes: List of (x, y, w, h, text, conf) tuples

        Returns:
            Tuple of (x, y, width, height) for the combined bounding box
        """
        xs = [b[0] for b in group_boxes]
        ys = [b[1] for b in group_boxes]
        x2s = [b[0] + b[2] for b in group_boxes]
        y2s = [b[1] + b[3] for b in group_boxes]

        return (min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys))

    def _group_into_lines(self, detections: List, image_shape: Tuple[int, int, int]) -> Tuple[List, List, List]:
        """Group detected text regions into lines using DBSCAN clustering.

        This method orchestrates the line grouping process:
        1. Convert detections to bounding boxes
        2. Cluster boxes by y-center (same row)
        3. Split by column gaps within each row
        4. Calculate final line bounding boxes

        Args:
            detections: List of EasyOCR detections (pts, text, confidence)
            image_shape: Shape of the image (h, w, channels)

        Returns:
            Tuple of (sorted_lines, line_confidences, easyocr_line_texts)
            - sorted_lines: List of (line_bbox, group_boxes) tuples
            - line_confidences: List of average confidence per line
            - easyocr_line_texts: List of concatenated EasyOCR text per line
        """
        h_img, w_img = image_shape[:2]

        # Get bounding boxes with confidence
        boxes = []
        for det in detections:
            pts = np.array(det[0])
            x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
            cy = y + h // 2
            boxes.append((x, y, w, h, cy, det[1], det[2]))  # det[2] is confidence

        # Cluster by y-center to group boxes on the same line
        labels = self._cluster_boxes_by_y(boxes)

        # Group boxes by line label
        lines = {}
        for i, (x, y, w, h, cy, text, conf) in enumerate(boxes):
            label = labels[i]
            if label not in lines:
                lines[label] = []
            lines[label].append((x, y, w, h, text, conf))

        # Calculate column gap threshold
        column_gap_threshold = w_img * COLUMN_GAP_RATIO

        # Sort lines top to bottom, split by large x-gaps (columns)
        sorted_lines = []
        line_confidences = []
        easyocr_line_texts = []

        for label in sorted(lines.keys(), key=lambda l: np.mean([b[1] for b in lines[l]])):
            line_boxes = sorted(lines[label], key=lambda b: b[0])

            # Split line by column gaps
            split_groups = self._split_by_column_gap(line_boxes, column_gap_threshold)

            # Process each split group as a separate line
            for group_boxes in split_groups:
                line_bbox = self._calculate_line_bbox(group_boxes)
                sorted_lines.append((line_bbox, group_boxes))

                # Concatenate EasyOCR text from all boxes (b[4] is text)
                easyocr_text = " ".join([b[4] for b in group_boxes])
                easyocr_line_texts.append(easyocr_text)

                # Average confidence for the line (b[5] is confidence)
                avg_conf = np.mean([b[5] for b in group_boxes])
                line_confidences.append(avg_conf)

        return sorted_lines, line_confidences, easyocr_line_texts

    def _crop_and_preprocess(self, image: np.ndarray, sorted_lines: List) -> Tuple[List[Image.Image], List]:
        """Crop line regions and preprocess for recognition.

        Args:
            image: BGR numpy array of the image
            sorted_lines: List of (line_bbox, group_boxes) tuples from _group_into_lines

        Returns:
            Tuple of (preprocessed PIL crops, crop_bboxes with raw crops)
            - crops: List of preprocessed PIL Images ready for TrOCR
            - crop_bboxes: List of (x, y, w, h, raw_crop) tuples for each line
        """
        h_img, w_img = image.shape[:2]

        # Crop lines with adaptive padding based on text size
        raw_crops = []
        crop_bboxes = []

        for line_bbox, _ in sorted_lines:
            x, y, w, h = line_bbox

            # Calculate adaptive padding based on line height
            padding_x, padding_y = calculate_adaptive_padding(h, w_img)

            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)
            x2 = min(w_img, x + w + padding_x)
            y2 = min(h_img, y + h + padding_y)

            crop = image[y1:y2, x1:x2]
            raw_crops.append(crop)
            crop_bboxes.append((x1, y1, x2-x1, y2-y1, crop))

        # Parallel preprocessing of all crops
        crops = [None] * len(raw_crops)
        crop_tasks = [(i, raw_crops[i], 'auto') for i in range(len(raw_crops))]

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(self.preprocess_crop_parallel, task): task[0] for task in crop_tasks}
            for future in as_completed(futures):
                idx, pil_crop = future.result()
                crops[idx] = pil_crop

        return crops, crop_bboxes

    def _recognize_text(self, crops: List[Image.Image], crop_bboxes: List) -> List[str]:
        """Recognize text from preprocessed crops with retry logic for empty results.

        Args:
            crops: List of preprocessed PIL Images
            crop_bboxes: List of (x, y, w, h, raw_crop) tuples for retry preprocessing

        Returns:
            List of recognized text strings, one per crop
        """
        # Batch recognize
        texts = self.recognize_batch(crops)

        # Multi-pass retry for empty results
        # If TrOCR returns empty, try with different preprocessing styles
        retry_styles = ['light', 'heavy', 'none']  # 'auto' was already tried
        for i, text in enumerate(texts):
            if not text or not text.strip():
                # Get the raw crop for this line
                x1, y1, w, h, raw_crop = crop_bboxes[i]

                # Try different preprocessing styles
                for style in retry_styles:
                    if style == 'none':
                        # No preprocessing - use raw image converted to PIL
                        retry_crop = Image.fromarray(cv2.cvtColor(raw_crop, cv2.COLOR_BGR2RGB))
                    else:
                        retry_crop = preprocess_crop(raw_crop, style=style)

                    retry_texts = self.recognize_batch([retry_crop])
                    if retry_texts and retry_texts[0] and retry_texts[0].strip():
                        texts[i] = retry_texts[0]
                        print(f"[Retry] Line {i}: Empty result recovered with style='{style}' -> '{retry_texts[0]}'")
                        break

        return texts

    @staticmethod
    def _looks_like_garbage(txt: str) -> bool:
        """Check if text looks like OCR garbage (special chars, no vowels, etc).

        Args:
            txt: Text string to check

        Returns:
            True if text appears to be garbage, False otherwise
        """
        if not txt or len(txt) < 2:
            return True

        # Clean punctuation from ends for analysis
        txt_clean = txt.strip().rstrip('.,;:!?')

        # Extended special char list
        special_chars = sum(1 for c in txt_clean if c in '|;{}[]<>$+@#%^&*()')
        alpha_chars = sum(1 for c in txt_clean if c.isalpha())

        # More than 15% special chars = garbage
        if alpha_chars == 0 or (special_chars / max(1, len(txt_clean)) > 0.15):
            return True

        # Check for fragmented/nonsense words (no vowels or too short)
        words = [w for w in txt_clean.replace('-', ' ').split() if w.strip()]
        vowels = set('aeiouAEIOU')
        nonsense_words = 0
        real_words = 0

        for word in words:
            word_clean = ''.join(c for c in word if c.isalpha())
            if len(word_clean) == 0:
                continue  # Skip punctuation-only "words"
            real_words += 1
            # Words with no vowels are suspicious (unless very short like "2%")
            if len(word_clean) > 2 and not any(c in vowels for c in word_clean):
                nonsense_words += 1

        # If more than 50% of real words are nonsense, it's garbage
        if real_words > 0 and nonsense_words / real_words > 0.5:
            return True

        return False

    def _filter_and_correct_results(
        self,
        texts: List[str],
        crop_bboxes: List,
        line_confidences: List[float],
        easyocr_line_texts: List[str],
        lines_to_verify: set,
        verify: bool
    ) -> Tuple[List[dict], float, int]:
        """Apply filtering and corrections to recognized text.

        Args:
            texts: List of recognized text strings
            crop_bboxes: List of (x, y, w, h, raw_crop) tuples
            line_confidences: List of confidence scores per line
            easyocr_line_texts: List of EasyOCR fallback text per line
            lines_to_verify: Set of line indices to verify with vision
            verify: Whether verification is enabled

        Returns:
            Tuple of (results list, total verification time, verified count)
        """
        results = []
        t_verify = 0
        verified_count = 0

        for i, (bbox_info, text) in enumerate(zip(crop_bboxes, texts)):
            x, y, w, h, crop_img = bbox_info
            confidence = line_confidences[i]
            easyocr_text = easyocr_line_texts[i] if i < len(easyocr_line_texts) else ""

            # Compare TrOCR vs EasyOCR - only use EasyOCR when it's clearly better
            trocr_is_garbage = self._looks_like_garbage(text)
            easyocr_is_garbage = self._looks_like_garbage(easyocr_text)

            # Only use EasyOCR if TrOCR is garbage AND EasyOCR is clearly better
            use_easyocr = False
            if not text and easyocr_text and not easyocr_is_garbage:
                use_easyocr = True
                print(f"[Fallback] Line {i}: TrOCR empty, using EasyOCR '{easyocr_text}'")
            elif trocr_is_garbage and not easyocr_is_garbage:
                use_easyocr = True
                print(f"[Fallback] Line {i}: TrOCR garbage '{text}', using EasyOCR '{easyocr_text}'")

            if use_easyocr:
                text = easyocr_text

            # Filter strikethrough
            if self.strike_detector and self.filter_strike:
                filtered, had_strike, ratio = self.filter_strike(
                    crop_img, text, self.strike_detector
                )
                if had_strike:
                    text = filtered

            # Selective vision verification
            verified = False
            if i in lines_to_verify:
                t_v0 = time.time()
                original = text
                text = self.verify_with_vision(crop_img, text)
                t_v1 = time.time()
                t_verify += (t_v1 - t_v0)
                verified_count += 1
                if text != original:
                    print(f"[Verify] Line {i}: '{original}' -> '{text}'")
                verified = True

            # Apply grocery item correction (conservative high-threshold matching)
            corrected = False
            if self.correct_grocery:
                original = text
                # Use conservative threshold of 0.88 - only correct high-confidence matches
                text, corrected, score = self.correct_grocery(text, threshold=0.88)
                if corrected:
                    print(f"[Correct] Line {i}: '{original}' -> '{text}' (match={score:.2f})")

            results.append({
                "index": i,
                "bbox": [x, y, w, h],
                "text": text.strip(),
                "confidence": round(confidence, 3),
                "verified": verified,
                "corrected": corrected
            })

        return results, t_verify, verified_count

    def process_image(self, image_path: str, verify: bool = False, max_verify: int = 2) -> dict:
        """Full OCR pipeline on an image.

        Args:
            image_path: Path to image file
            verify: Enable selective LLaVA verification on low-confidence lines
            max_verify: Maximum number of lines to verify (default: 2)
        """
        t0 = time.time()

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Could not load: {image_path}"}

        # Detect text regions
        detections, detection_time = self._detect_text_regions(image)

        if not detections:
            return {
                "image_path": image_path,
                "lines": [],
                "full_text": "",
                "timing": {"total": time.time() - t0}
            }

        # Group detections into lines using DBSCAN clustering
        sorted_lines, line_confidences, easyocr_line_texts = self._group_into_lines(
            detections, image.shape
        )

        # Crop and preprocess line regions
        crops, crop_bboxes = self._crop_and_preprocess(image, sorted_lines)
        t_grouping = time.time()

        # Recognize text with retry logic for empty results
        texts = self._recognize_text(crops, crop_bboxes)
        t_recognition = time.time()

        # Determine which lines to verify (lowest confidence)
        lines_to_verify = set()
        if verify and max_verify > 0:
            # Get indices of lowest confidence lines
            sorted_by_conf = sorted(range(len(line_confidences)),
                                   key=lambda i: line_confidences[i])
            lines_to_verify = set(sorted_by_conf[:max_verify])
            print(f"[Verify] Will verify {len(lines_to_verify)} lowest confidence lines")

        # Apply filtering and corrections
        results, t_verify, verified_count = self._filter_and_correct_results(
            texts, crop_bboxes, line_confidences, easyocr_line_texts,
            lines_to_verify, verify
        )
        t_filtering = time.time()

        timing = {
            "detection": round(detection_time, 2),
            "grouping": round(t_grouping - t0 - detection_time, 2),
            "recognition": round(t_recognition - t_grouping, 2),
            "filtering": round(t_filtering - t_recognition - t_verify, 2),
            "total": round(t_filtering - t0, 2)
        }
        if verify:
            timing["verification"] = round(t_verify, 2)
            timing["verified_lines"] = verified_count

        return {
            "image_path": image_path,
            "lines": results,
            "full_text": "\n".join(r["text"] for r in results),
            "timing": timing
        }

    def serve_socket(self):
        """Run as Unix socket server for fastest IPC.

        Security features:
        - Unix socket (AF_UNIX) ensures localhost-only access
        - Rate limiting prevents DoS attacks (100 req/min default)
        - Input validation prevents oversized/malformed requests
        - Path validation prevents directory traversal
        """
        import os

        # Remove old socket
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)

        self.load_models()

        # Initialize rate limiter
        rate_limiter = RateLimiter()

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        server.listen(5)

        print(f"[Server] Listening on {SOCKET_PATH}")
        print(f"[Server] Rate limit: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s")
        print("[Server] Ready for requests!")

        while True:
            conn, _ = server.accept()
            try:
                # Check rate limit before processing
                rate_limiter.check_rate_limit()

                # Read message length
                raw_len = conn.recv(4)
                if not raw_len:
                    continue
                msg_len = struct.unpack('>I', raw_len)[0]

                # Security: Reject oversized messages early
                if msg_len > MAX_REQUEST_SIZE:
                    raise InputValidationError(
                        f"Message too large: {msg_len} bytes (max: {MAX_REQUEST_SIZE})"
                    )

                # Read message
                data = b''
                while len(data) < msg_len:
                    chunk = conn.recv(min(4096, msg_len - len(data)))
                    if not chunk:
                        break
                    data += chunk

                request = json.loads(data.decode())

                # Validate all input fields
                validate_request_input(request, len(data))

                # Process
                if request.get('action') == 'ocr':
                    # Validate path before processing (security check)
                    validated_path = validate_image_path(request['image_path'])
                    result = self.process_image(
                        str(validated_path),
                        verify=request.get('verify', False),
                        max_verify=request.get('max_verify', 2)
                    )
                elif request.get('action') == 'ping':
                    result = {"status": "ok"}
                else:
                    result = {"error": "Unknown action"}

                # Send response
                response = json.dumps(result).encode()
                conn.sendall(struct.pack('>I', len(response)) + response)

            except RateLimitError as e:
                print(f"[Server] Rate limit exceeded: {e}")
                error = json.dumps({"error": f"Rate limit exceeded: {e}"}).encode()
                conn.sendall(struct.pack('>I', len(error)) + error)
            except InputValidationError as e:
                print(f"[Server] Input validation failed: {e}")
                error = json.dumps({"error": f"Validation error: {e}"}).encode()
                conn.sendall(struct.pack('>I', len(error)) + error)
            except json.JSONDecodeError as e:
                print(f"[Server] Invalid JSON request: {e}")
                error = json.dumps({"error": f"Invalid JSON: {e}"}).encode()
                conn.sendall(struct.pack('>I', len(error)) + error)
            except PathSecurityError as e:
                print(f"[Server] Security violation: {e}")
                error = json.dumps({"error": f"Security error: {e}"}).encode()
                conn.sendall(struct.pack('>I', len(error)) + error)
            except (KeyError, TypeError) as e:
                print(f"[Server] Invalid request format: {e}")
                error = json.dumps({"error": f"Invalid request: {e}"}).encode()
                conn.sendall(struct.pack('>I', len(error)) + error)
            except FileNotFoundError as e:
                print(f"[Server] File not found: {e}")
                error = json.dumps({"error": f"File not found: {e}"}).encode()
                conn.sendall(struct.pack('>I', len(error)) + error)
            except Exception as e:
                # Intentional catch-all as last-resort handler for unexpected errors.
                # All known error types are caught above; this prevents server crashes
                # from unhandled exceptions while ensuring full tracebacks are logged.
                print(f"[Server] Unexpected error: {e}")
                traceback.print_exc()
                error = json.dumps({"error": str(e)}).encode()
                conn.sendall(struct.pack('>I', len(error)) + error)
            finally:
                conn.close()


def send_request(action: str, **kwargs) -> dict:
    """Send a request to the running OCR server via Unix socket.

    Args:
        action: The action to perform ('ocr', 'ping')
        **kwargs: Additional parameters for the action:
            - image_path: Path to image file (for 'ocr' action)
            - verify: Enable vision verification (for 'ocr' action)
            - max_verify: Maximum lines to verify (for 'ocr' action)

    Returns:
        dict: Server response containing OCR results or status

    Raises:
        ConnectionRefusedError: If the server is not running
        socket.error: If there's a socket communication error
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(SOCKET_PATH)

    request = {"action": action, **kwargs}
    data = json.dumps(request).encode()
    sock.sendall(struct.pack('>I', len(data)) + data)

    # Read response
    raw_len = sock.recv(4)
    msg_len = struct.unpack('>I', raw_len)[0]

    response = b''
    while len(response) < msg_len:
        chunk = sock.recv(min(4096, msg_len - len(response)))
        response += chunk

    sock.close()
    return json.loads(response.decode())


def main():
    """Main entry point for the OCR server CLI.

    Commands:
        serve: Start the OCR server (keeps TrOCR model loaded)
        process <image_path>: Process an image using the running server
            --verify: Enable LLaVA vision verification
            --max-verify N: Maximum lines to verify (default: 2)
        ping: Check if server is running

    Example:
        python ocr_server.py serve &
        python ocr_server.py process image.jpg --verify
    """
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == 'serve':
        server = OCRServer()
        server.serve_socket()

    elif cmd == 'process':
        if len(sys.argv) < 3:
            print("Usage: python ocr_server.py process <image_path> [--verify] [--max-verify N]")
            return

        image_path = sys.argv[2]

        # Parse optional flags
        verify = '--verify' in sys.argv
        max_verify = 2
        if '--max-verify' in sys.argv:
            idx = sys.argv.index('--max-verify')
            if idx + 1 < len(sys.argv):
                max_verify = int(sys.argv[idx + 1])

        t0 = time.time()
        result = send_request(
            'ocr',
            image_path=str(Path(image_path).absolute()),
            verify=verify,
            max_verify=max_verify
        )
        t1 = time.time()

        print(json.dumps(result, indent=2))
        print(f"\n[Client] Total time: {t1-t0:.2f}s")
        if verify:
            print(f"[Client] Verified {result.get('timing', {}).get('verified_lines', 0)} lines with LLaVA")

    elif cmd == 'ping':
        result = send_request('ping')
        print(result)

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == '__main__':
    main()
