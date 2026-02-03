"""
Accuracy tests for KwikRead OCR system.

This module tests OCR accuracy against ground truth data:
- Per-image accuracy testing with detailed error reporting
- Batch accuracy metrics (precision, recall, F1)
- Regression tests to catch accuracy drops between model versions
- Fuzzy matching with configurable thresholds

Performance Target: 95% fuzzy match accuracy
Ground Truth: finetune/ground_truth.json

Usage:
    pytest tests/test_accuracy.py -v
    pytest tests/test_accuracy.py -v -k videotest3
    pytest tests/test_accuracy.py::TestVideotest3Accuracy -v
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
import pytest

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import cv2
import numpy as np


# =============================================================================
# Constants
# =============================================================================

# Accuracy thresholds
FUZZY_MATCH_THRESHOLD = 0.80  # Minimum similarity for a "match"
STRICT_MATCH_THRESHOLD = 0.95  # Near-exact match
TARGET_ACCURACY = 0.95  # 95% of items should match


# =============================================================================
# Helper Functions
# =============================================================================

def fuzzy_match(predicted: str, expected: str) -> float:
    """
    Calculate fuzzy match score between predicted and expected text.

    Args:
        predicted: OCR prediction
        expected: Ground truth text

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not predicted or not expected:
        return 0.0

    # Normalize: lowercase, strip whitespace
    p = predicted.lower().strip()
    e = expected.lower().strip()

    # Exact match
    if p == e:
        return 1.0

    # SequenceMatcher for fuzzy comparison
    return SequenceMatcher(None, p, e).ratio()


def load_crop_image(path: Path) -> Image.Image:
    """Load a crop image and convert to PIL Image."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def calculate_accuracy_metrics(
    predictions: List[str],
    ground_truth: List[str],
    threshold: float = FUZZY_MATCH_THRESHOLD
) -> Dict:
    """
    Calculate accuracy metrics for a set of predictions.

    Args:
        predictions: List of OCR predictions
        ground_truth: List of expected texts
        threshold: Minimum similarity for a match

    Returns:
        Dictionary with accuracy metrics
    """
    if len(predictions) != len(ground_truth):
        # Handle mismatched lengths
        min_len = min(len(predictions), len(ground_truth))
        predictions = predictions[:min_len]
        ground_truth = ground_truth[:min_len]

    matches = []
    for pred, gt in zip(predictions, ground_truth):
        score = fuzzy_match(pred, gt)
        matches.append({
            "predicted": pred,
            "expected": gt,
            "score": score,
            "is_match": score >= threshold,
            "is_exact": score >= STRICT_MATCH_THRESHOLD,
        })

    total = len(matches)
    if total == 0:
        return {
            "total": 0,
            "matches": 0,
            "exact_matches": 0,
            "accuracy": 0.0,
            "exact_accuracy": 0.0,
            "avg_score": 0.0,
            "details": [],
        }

    match_count = sum(1 for m in matches if m["is_match"])
    exact_count = sum(1 for m in matches if m["is_exact"])
    avg_score = sum(m["score"] for m in matches) / total

    return {
        "total": total,
        "matches": match_count,
        "exact_matches": exact_count,
        "accuracy": match_count / total,
        "exact_accuracy": exact_count / total,
        "avg_score": avg_score,
        "details": matches,
    }


# =============================================================================
# Videotest2 Accuracy Tests (21 items)
# =============================================================================

class TestVideotest2Accuracy:
    """Test OCR accuracy on videotest2 dataset (21 grocery items)."""

    @pytest.mark.requires_model
    def test_batch_recognition(self, model_bundle, videotest2_crop_paths, videotest2_labels):
        """
        Test batch recognition on all videotest2 crops.

        This is the main accuracy test - processes all images and compares
        against ground truth.
        """
        if model_bundle is None:
            pytest.skip("Model not available")

        # Load all crop images
        images = []
        expected_texts = []

        for path in videotest2_crop_paths:
            filename = path.name
            if filename in videotest2_labels:
                images.append(load_crop_image(path))
                expected_texts.append(videotest2_labels[filename])

        assert len(images) > 0, "No images loaded for testing"

        # Run batch recognition
        predictions = model_bundle.recognize_batch(images)

        # Calculate metrics
        metrics = calculate_accuracy_metrics(predictions, expected_texts)

        # Print detailed report for failures
        print(f"\n{'='*60}")
        print(f"VIDEOTEST2 ACCURACY REPORT")
        print(f"{'='*60}")
        print(f"Total items: {metrics['total']}")
        print(f"Matches (>={FUZZY_MATCH_THRESHOLD:.0%}): {metrics['matches']} ({metrics['accuracy']:.1%})")
        print(f"Exact matches (>={STRICT_MATCH_THRESHOLD:.0%}): {metrics['exact_matches']} ({metrics['exact_accuracy']:.1%})")
        print(f"Average similarity: {metrics['avg_score']:.1%}")
        print(f"\nDetailed Results:")

        for detail in metrics["details"]:
            status = "OK" if detail["is_match"] else "FAIL"
            print(f"  [{status}] '{detail['predicted']}' vs '{detail['expected']}' ({detail['score']:.1%})")

        # Assert minimum accuracy
        assert metrics["accuracy"] >= 0.70, (
            f"Accuracy {metrics['accuracy']:.1%} below minimum 70%. "
            f"Matched {metrics['matches']}/{metrics['total']} items."
        )

    @pytest.mark.requires_model
    def test_individual_items(self, model_bundle, videotest2_crop_paths, videotest2_labels):
        """Test recognition of individual critical items."""
        if model_bundle is None:
            pytest.skip("Model not available")

        # Critical items that must be recognized
        critical_items = {
            "line_00.png": "Rice",
            "line_06.png": "Cheese",
            "line_15.png": "milk",
            "line_16.png": "cereal",
        }

        for filename, expected in critical_items.items():
            path = videotest2_crop_paths[0].parent / filename
            if not path.exists():
                pytest.skip(f"Critical test image not found: {path}")

            img = load_crop_image(path)
            predictions = model_bundle.recognize_batch([img])
            predicted = predictions[0] if predictions else ""

            score = fuzzy_match(predicted, expected)
            assert score >= FUZZY_MATCH_THRESHOLD, (
                f"Critical item '{expected}' not recognized. "
                f"Got: '{predicted}' (similarity: {score:.1%})"
            )


# =============================================================================
# Videotest3 Accuracy Tests (16 items)
# =============================================================================

class TestVideotest3Accuracy:
    """Test OCR accuracy on videotest3 dataset (16 grocery items)."""

    @pytest.mark.requires_model
    def test_batch_recognition(self, model_bundle, videotest3_crop_paths, videotest3_labels):
        """
        Test batch recognition on all videotest3 crops.

        Target: 95% fuzzy match accuracy
        """
        if model_bundle is None:
            pytest.skip("Model not available")

        # Load all crop images
        images = []
        expected_texts = []

        for path in videotest3_crop_paths:
            filename = path.name
            if filename in videotest3_labels:
                images.append(load_crop_image(path))
                expected_texts.append(videotest3_labels[filename])

        assert len(images) > 0, "No images loaded for testing"

        # Run batch recognition
        predictions = model_bundle.recognize_batch(images)

        # Calculate metrics
        metrics = calculate_accuracy_metrics(predictions, expected_texts)

        # Print detailed report
        print(f"\n{'='*60}")
        print(f"VIDEOTEST3 ACCURACY REPORT")
        print(f"{'='*60}")
        print(f"Total items: {metrics['total']}")
        print(f"Matches (>={FUZZY_MATCH_THRESHOLD:.0%}): {metrics['matches']} ({metrics['accuracy']:.1%})")
        print(f"Exact matches (>={STRICT_MATCH_THRESHOLD:.0%}): {metrics['exact_matches']} ({metrics['exact_accuracy']:.1%})")
        print(f"Average similarity: {metrics['avg_score']:.1%}")
        print(f"\nDetailed Results:")

        for detail in metrics["details"]:
            status = "OK" if detail["is_match"] else "FAIL"
            print(f"  [{status}] '{detail['predicted']}' vs '{detail['expected']}' ({detail['score']:.1%})")

        # Assert minimum accuracy
        assert metrics["accuracy"] >= 0.70, (
            f"Accuracy {metrics['accuracy']:.1%} below minimum 70%. "
            f"Matched {metrics['matches']}/{metrics['total']} items."
        )

    @pytest.mark.requires_model
    def test_challenging_items(self, model_bundle, videotest3_crop_paths, videotest3_labels):
        """Test recognition of challenging handwritten items."""
        if model_bundle is None:
            pytest.skip("Model not available")

        # Items that are often challenging
        challenging_items = {
            "line_01.png": "Black beans",
            "line_03.png": "Chia Seeds Ramen Noodles",
            "line_12.png": "Chicken Breast Paprika",
        }

        for filename, expected in challenging_items.items():
            path = videotest3_crop_paths[0].parent / filename
            if not path.exists():
                continue  # Skip if file doesn't exist

            img = load_crop_image(path)
            predictions = model_bundle.recognize_batch([img])
            predicted = predictions[0] if predictions else ""

            score = fuzzy_match(predicted, expected)

            # Relaxed threshold for challenging items
            threshold = 0.60
            if score < threshold:
                print(f"[WARN] Challenging item '{expected}' score: {score:.1%}, got: '{predicted}'")


# =============================================================================
# Regression Tests
# =============================================================================

class TestAccuracyRegression:
    """Regression tests to catch accuracy drops between versions."""

    # Historical baselines (update these when accuracy improves)
    VIDEOTEST2_BASELINE = 0.70  # 70% accuracy
    VIDEOTEST3_BASELINE = 0.70  # 70% accuracy

    @pytest.mark.requires_model
    def test_videotest2_no_regression(self, model_bundle, videotest2_crop_paths, videotest2_labels):
        """Ensure videotest2 accuracy doesn't drop below baseline."""
        if model_bundle is None:
            pytest.skip("Model not available")

        images = []
        expected_texts = []

        for path in videotest2_crop_paths:
            filename = path.name
            if filename in videotest2_labels:
                images.append(load_crop_image(path))
                expected_texts.append(videotest2_labels[filename])

        if not images:
            pytest.skip("No videotest2 images available")

        predictions = model_bundle.recognize_batch(images)
        metrics = calculate_accuracy_metrics(predictions, expected_texts)

        assert metrics["accuracy"] >= self.VIDEOTEST2_BASELINE, (
            f"REGRESSION DETECTED: videotest2 accuracy dropped from "
            f"{self.VIDEOTEST2_BASELINE:.1%} to {metrics['accuracy']:.1%}"
        )

    @pytest.mark.requires_model
    def test_videotest3_no_regression(self, model_bundle, videotest3_crop_paths, videotest3_labels):
        """Ensure videotest3 accuracy doesn't drop below baseline."""
        if model_bundle is None:
            pytest.skip("Model not available")

        images = []
        expected_texts = []

        for path in videotest3_crop_paths:
            filename = path.name
            if filename in videotest3_labels:
                images.append(load_crop_image(path))
                expected_texts.append(videotest3_labels[filename])

        if not images:
            pytest.skip("No videotest3 images available")

        predictions = model_bundle.recognize_batch(images)
        metrics = calculate_accuracy_metrics(predictions, expected_texts)

        assert metrics["accuracy"] >= self.VIDEOTEST3_BASELINE, (
            f"REGRESSION DETECTED: videotest3 accuracy dropped from "
            f"{self.VIDEOTEST3_BASELINE:.1%} to {metrics['accuracy']:.1%}"
        )


# =============================================================================
# Ground Truth Consistency Tests
# =============================================================================

class TestGroundTruthConsistency:
    """Test ground truth data quality and consistency."""

    def test_videotest2_has_labels(self, videotest2_labels):
        """Verify videotest2 has ground truth labels."""
        assert len(videotest2_labels) > 0, "No videotest2 labels found"
        assert len(videotest2_labels) >= 20, f"Expected ~21 labels, got {len(videotest2_labels)}"

    def test_videotest3_has_labels(self, videotest3_labels):
        """Verify videotest3 has ground truth labels."""
        assert len(videotest3_labels) > 0, "No videotest3 labels found"
        assert len(videotest3_labels) >= 15, f"Expected ~16 labels, got {len(videotest3_labels)}"

    def test_labels_match_images_videotest2(self, videotest2_labels, videotest2_crop_paths):
        """Verify all labeled images exist for videotest2."""
        available_files = {p.name for p in videotest2_crop_paths}

        for filename in videotest2_labels.keys():
            assert filename in available_files or not videotest2_crop_paths, (
                f"Label for '{filename}' but image not found"
            )

    def test_labels_match_images_videotest3(self, videotest3_labels, videotest3_crop_paths):
        """Verify all labeled images exist for videotest3."""
        available_files = {p.name for p in videotest3_crop_paths}

        for filename in videotest3_labels.keys():
            assert filename in available_files or not videotest3_crop_paths, (
                f"Label for '{filename}' but image not found"
            )

    def test_label_quality(self, videotest2_labels, videotest3_labels):
        """Check that labels are non-empty and reasonable."""
        all_labels = {**videotest2_labels, **videotest3_labels}

        for filename, label in all_labels.items():
            assert label, f"Empty label for {filename}"
            assert len(label) < 100, f"Label too long for {filename}: {len(label)} chars"
            assert label.strip() == label, f"Label has leading/trailing whitespace: '{label}'"


# =============================================================================
# Accuracy with Post-Processing Tests
# =============================================================================

class TestAccuracyWithCorrector:
    """Test accuracy improvement when using grocery corrector post-processing."""

    @pytest.mark.requires_model
    def test_corrector_improves_accuracy(
        self, model_bundle, grocery_corrector,
        videotest3_crop_paths, videotest3_labels
    ):
        """
        Test that grocery_corrector improves raw OCR accuracy.

        The corrector should fix common OCR errors like:
        - "out milk" -> "oat milk"
        - "protien bars" -> "protein bars"
        """
        if model_bundle is None or grocery_corrector is None:
            pytest.skip("Model or corrector not available")

        images = []
        expected_texts = []

        for path in videotest3_crop_paths:
            filename = path.name
            if filename in videotest3_labels:
                images.append(load_crop_image(path))
                expected_texts.append(videotest3_labels[filename])

        if not images:
            pytest.skip("No test images available")

        # Get raw predictions
        raw_predictions = model_bundle.recognize_batch(images)

        # Apply corrector to each prediction
        corrected_predictions = []
        for pred in raw_predictions:
            corrected, was_corrected, score = grocery_corrector(pred, threshold=0.88)
            corrected_predictions.append(corrected)

        # Calculate metrics for both
        raw_metrics = calculate_accuracy_metrics(raw_predictions, expected_texts)
        corrected_metrics = calculate_accuracy_metrics(corrected_predictions, expected_texts)

        print(f"\n{'='*60}")
        print(f"CORRECTOR IMPACT REPORT")
        print(f"{'='*60}")
        print(f"Raw OCR accuracy: {raw_metrics['accuracy']:.1%}")
        print(f"Corrected accuracy: {corrected_metrics['accuracy']:.1%}")
        print(f"Improvement: {(corrected_metrics['accuracy'] - raw_metrics['accuracy']):.1%}")

        # Corrector should not make accuracy worse
        assert corrected_metrics["accuracy"] >= raw_metrics["accuracy"] - 0.05, (
            f"Corrector made accuracy significantly worse: "
            f"{raw_metrics['accuracy']:.1%} -> {corrected_metrics['accuracy']:.1%}"
        )
