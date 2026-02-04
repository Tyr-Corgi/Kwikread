"""
Pytest configuration and shared fixtures for KwikRead OCR tests.

This module provides:
- Shared fixtures for model loading (expensive - loaded once per session)
- Ground truth data fixtures
- Test image path fixtures
- Helper utilities for tests

Usage:
    pytest tests/ -v
    pytest tests/test_accuracy.py -v
    pytest tests/test_performance.py -v --benchmark
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pytest

# Add parent directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Path Constants
# =============================================================================

FINETUNE_DIR = PROJECT_ROOT / "finetune"
GROUND_TRUTH_PATH = FINETUNE_DIR / "ground_truth.json"
VIDEOTEST2_CROPS_DIR = PROJECT_ROOT / "videotest2_v15" / "crops" / "videotest2_frame"
VIDEOTEST3_CROPS_DIR = FINETUNE_DIR / "crops" / "videotest3"

# Model paths (matching ocr_server.py)
MODEL_V3_PATH = FINETUNE_DIR / "model_v3" / "final"
MODEL_V2_PATH = FINETUNE_DIR / "model_v2" / "final"
MODEL_V1_PATH = FINETUNE_DIR / "model" / "final"
BASE_MODEL_NAME = "microsoft/trocr-base-handwritten"

# Performance targets
PERFORMANCE_TARGET_SECONDS = 6.0  # <6 seconds for 16 images
ACCURACY_TARGET = 0.95  # 95% fuzzy match accuracy


# =============================================================================
# Ground Truth Fixture
# =============================================================================

@pytest.fixture(scope="session")
def ground_truth_data() -> Dict:
    """
    Load ground truth data from JSON file.

    Returns:
        Dictionary with videotest2 and videotest3 ground truth labels.
    """
    if not GROUND_TRUTH_PATH.exists():
        pytest.skip(f"Ground truth file not found: {GROUND_TRUTH_PATH}")

    with open(GROUND_TRUTH_PATH, "r") as f:
        data = json.load(f)

    return data


@pytest.fixture(scope="session")
def videotest2_labels(ground_truth_data) -> Dict[str, str]:
    """Get videotest2 ground truth labels (filename -> text)."""
    return ground_truth_data.get("videotest2", {}).get("labels", {})


@pytest.fixture(scope="session")
def videotest3_labels(ground_truth_data) -> Dict[str, str]:
    """Get videotest3 ground truth labels (filename -> text)."""
    return ground_truth_data.get("videotest3", {}).get("labels", {})


@pytest.fixture(scope="session")
def videotest2_items(videotest2_labels) -> List[str]:
    """Get ordered list of videotest2 ground truth items."""
    # Sort by line number to ensure correct order
    sorted_items = sorted(videotest2_labels.items(),
                         key=lambda x: int(x[0].replace("line_", "").replace(".png", "")))
    return [text for _, text in sorted_items]


@pytest.fixture(scope="session")
def videotest3_items(videotest3_labels) -> List[str]:
    """Get ordered list of videotest3 ground truth items."""
    sorted_items = sorted(videotest3_labels.items(),
                         key=lambda x: int(x[0].replace("line_", "").replace(".png", "")))
    return [text for _, text in sorted_items]


# =============================================================================
# Test Image Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def videotest2_crop_paths() -> List[Path]:
    """Get sorted list of videotest2 crop image paths."""
    if not VIDEOTEST2_CROPS_DIR.exists():
        pytest.skip(f"Videotest2 crops directory not found: {VIDEOTEST2_CROPS_DIR}")

    paths = sorted(VIDEOTEST2_CROPS_DIR.glob("line_*.png"),
                   key=lambda p: int(p.stem.replace("line_", "")))

    if not paths:
        pytest.skip(f"No crop images found in {VIDEOTEST2_CROPS_DIR}")

    return paths


@pytest.fixture(scope="session")
def videotest3_crop_paths() -> List[Path]:
    """Get sorted list of videotest3 crop image paths."""
    if not VIDEOTEST3_CROPS_DIR.exists():
        pytest.skip(f"Videotest3 crops directory not found: {VIDEOTEST3_CROPS_DIR}")

    paths = sorted(VIDEOTEST3_CROPS_DIR.glob("line_*.png"),
                   key=lambda p: int(p.stem.replace("line_", "")))

    if not paths:
        pytest.skip(f"No crop images found in {VIDEOTEST3_CROPS_DIR}")

    return paths


# =============================================================================
# Model Loading Fixture (Expensive - Session Scoped)
# =============================================================================

class ModelBundle:
    """Container for loaded model, processor, and device."""

    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device
        self.load_time_seconds: float = 0.0

    def recognize_batch(self, images: List) -> List[str]:
        """
        Batch recognize text from PIL images.

        Args:
            images: List of PIL Image objects

        Returns:
            List of recognized text strings
        """
        import torch

        if not images:
            return []

        with torch.inference_mode():
            pixel_values = self.processor(
                images=images,
                return_tensors='pt',
                padding=True
            ).pixel_values.to(self.device)

            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=20,
                use_cache=True
            )

            texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

        return texts


@pytest.fixture(scope="session")
def model_bundle() -> Optional[ModelBundle]:
    """
    Load TrOCR model and processor once per test session.

    This is an expensive operation (~5-10 seconds) so we cache it.

    Returns:
        ModelBundle with model, processor, and device info.
        None if models cannot be loaded.
    """
    start_time = time.time()

    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch
    except ImportError:
        pytest.skip("transformers or torch not installed")
        return None

    # Determine model path (prefer fine-tuned models)
    if MODEL_V3_PATH.exists():
        model_path = str(MODEL_V3_PATH)
    elif MODEL_V2_PATH.exists():
        model_path = str(MODEL_V2_PATH)
    elif MODEL_V1_PATH.exists():
        model_path = str(MODEL_V1_PATH)
    else:
        model_path = BASE_MODEL_NAME

    try:
        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
    except (OSError, RuntimeError, ValueError) as e:
        pytest.skip(f"Could not load model from {model_path}: {e}")
        return None

    # Select device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model.to(device)
    model.eval()

    bundle = ModelBundle(model, processor, device)
    bundle.load_time_seconds = time.time() - start_time

    return bundle


# =============================================================================
# Grocery Corrector Fixture
# =============================================================================

@pytest.fixture(scope="session")
def grocery_corrector():
    """
    Load the grocery corrector module.

    Returns:
        The correct_grocery_text function from grocery_corrector.py
    """
    try:
        from grocery_corrector import correct_grocery_text
        return correct_grocery_text
    except ImportError:
        pytest.skip("grocery_corrector module not available")
        return None


@pytest.fixture(scope="session")
def grocery_items():
    """
    Load the grocery items list for testing.

    Returns:
        Set of valid grocery item strings (lowercase)
    """
    try:
        from grocery_corrector import GROCERY_SET
        return GROCERY_SET
    except ImportError:
        pytest.skip("grocery_corrector module not available")
        return set()


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def fuzzy_matcher():
    """
    Provide fuzzy string matching function.

    Returns:
        Function that calculates similarity between two strings (0.0 to 1.0)
    """
    from difflib import SequenceMatcher

    def match(a: str, b: str) -> float:
        """Calculate fuzzy match ratio between two strings."""
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

    return match


@pytest.fixture
def timing_context():
    """
    Provide a context manager for timing operations.

    Usage:
        with timing_context() as timer:
            # do something
        elapsed = timer.elapsed
    """
    class Timer:
        def __init__(self):
            self.start_time = 0.0
            self.end_time = 0.0
            self.elapsed = 0.0

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()
            self.elapsed = self.end_time - self.start_time

    return Timer


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_model: marks tests that require model loading"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on their names."""
    for item in items:
        # Mark tests that use model_bundle as requiring model
        if "model_bundle" in item.fixturenames:
            item.add_marker(pytest.mark.requires_model)

        # Mark performance tests as slow
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)


# =============================================================================
# Test Session Info
# =============================================================================

def pytest_report_header(config):
    """Add project info to test report header."""
    return [
        "KwikRead OCR Test Suite",
        f"Project Root: {PROJECT_ROOT}",
        f"Ground Truth: {GROUND_TRUTH_PATH.exists()}",
        f"Videotest2 Crops: {VIDEOTEST2_CROPS_DIR.exists()}",
        f"Videotest3 Crops: {VIDEOTEST3_CROPS_DIR.exists()}",
    ]


# =============================================================================
# Shared Image Loading Fixture
# =============================================================================

@pytest.fixture(scope="session")
def load_crop_image():
    """
    Provide a function to load crop images as PIL Images.

    Returns:
        Function that takes a Path and returns a PIL Image.
    """
    from PIL import Image
    import cv2

    def _load_crop_image(path: Path) -> Image.Image:
        """Load a crop image and convert to PIL Image."""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return _load_crop_image
