"""
Performance tests for KwikRead OCR system.

This module tests OCR inference speed:
- Batch inference timing
- Target: <6 seconds for 16 images
- Current baseline: 3.79s (measured)
- Memory usage monitoring
- Throughput calculations

Usage:
    pytest tests/test_performance.py -v
    pytest tests/test_performance.py -v --benchmark
    pytest tests/test_performance.py::TestInferenceSpeed -v
"""

import sys
import time
from pathlib import Path
from typing import List
import pytest

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import cv2


# =============================================================================
# Performance Targets
# =============================================================================

# Maximum acceptable time for batch inference
PERFORMANCE_TARGET_16_IMAGES = 6.0  # seconds
PERFORMANCE_TARGET_PER_IMAGE = 0.5  # seconds per image (worst case)

# Current baseline (for regression detection)
CURRENT_BASELINE_16_IMAGES = 4.0  # seconds (with buffer)

# Minimum throughput
MIN_THROUGHPUT_IMAGES_PER_SEC = 2.5  # images per second


# =============================================================================
# Helper Functions
# =============================================================================

def load_crop_image(path: Path) -> Image.Image:
    """Load a crop image and convert to PIL Image."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def measure_inference_time(model_bundle, images: List[Image.Image], warmup: bool = True) -> float:
    """
    Measure inference time for a batch of images.

    Args:
        model_bundle: The loaded model bundle
        images: List of PIL Images
        warmup: Whether to run a warmup pass first

    Returns:
        Elapsed time in seconds
    """
    if warmup and len(images) > 0:
        # Warmup pass to ensure model is loaded into memory/GPU
        _ = model_bundle.recognize_batch([images[0]])

    start_time = time.perf_counter()
    _ = model_bundle.recognize_batch(images)
    end_time = time.perf_counter()

    return end_time - start_time


# =============================================================================
# Inference Speed Tests
# =============================================================================

class TestInferenceSpeed:
    """Test OCR inference speed meets performance targets."""

    @pytest.mark.slow
    @pytest.mark.requires_model
    def test_videotest3_under_6_seconds(self, model_bundle, videotest3_crop_paths):
        """
        Test that processing 16 videotest3 images takes <6 seconds.

        This is the primary performance target.
        """
        if model_bundle is None:
            pytest.skip("Model not available")

        # Load all images
        images = [load_crop_image(p) for p in videotest3_crop_paths]
        assert len(images) >= 16, f"Expected 16 images, got {len(images)}"

        # Measure inference time
        elapsed = measure_inference_time(model_bundle, images, warmup=True)

        # Report
        print(f"\n{'='*60}")
        print(f"PERFORMANCE REPORT: videotest3 (16 images)")
        print(f"{'='*60}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Target: <{PERFORMANCE_TARGET_16_IMAGES}s")
        print(f"Per image: {elapsed/len(images):.3f}s")
        print(f"Throughput: {len(images)/elapsed:.1f} images/sec")
        print(f"Status: {'PASS' if elapsed < PERFORMANCE_TARGET_16_IMAGES else 'FAIL'}")

        assert elapsed < PERFORMANCE_TARGET_16_IMAGES, (
            f"Performance target missed: {elapsed:.2f}s > {PERFORMANCE_TARGET_16_IMAGES}s"
        )

    @pytest.mark.slow
    @pytest.mark.requires_model
    def test_videotest2_performance(self, model_bundle, videotest2_crop_paths):
        """
        Test videotest2 (21 images) performance.

        Target: ~7.5 seconds (scaled from 6s for 16 images)
        """
        if model_bundle is None:
            pytest.skip("Model not available")

        images = [load_crop_image(p) for p in videotest2_crop_paths]

        # Scale target based on image count
        expected_target = PERFORMANCE_TARGET_16_IMAGES * (len(images) / 16)

        elapsed = measure_inference_time(model_bundle, images, warmup=True)

        print(f"\n{'='*60}")
        print(f"PERFORMANCE REPORT: videotest2 ({len(images)} images)")
        print(f"{'='*60}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Target: <{expected_target:.1f}s")
        print(f"Per image: {elapsed/len(images):.3f}s")
        print(f"Throughput: {len(images)/elapsed:.1f} images/sec")

        assert elapsed < expected_target, (
            f"Performance target missed: {elapsed:.2f}s > {expected_target:.1f}s"
        )

    @pytest.mark.requires_model
    def test_single_image_latency(self, model_bundle, videotest3_crop_paths):
        """Test single image inference latency."""
        if model_bundle is None:
            pytest.skip("Model not available")

        if not videotest3_crop_paths:
            pytest.skip("No test images available")

        # Load single image
        img = load_crop_image(videotest3_crop_paths[0])

        # Measure single inference (after warmup)
        _ = model_bundle.recognize_batch([img])  # warmup

        start = time.perf_counter()
        _ = model_bundle.recognize_batch([img])
        elapsed = time.perf_counter() - start

        print(f"\nSingle image latency: {elapsed*1000:.0f}ms")

        assert elapsed < PERFORMANCE_TARGET_PER_IMAGE, (
            f"Single image too slow: {elapsed:.2f}s > {PERFORMANCE_TARGET_PER_IMAGE}s"
        )


# =============================================================================
# Throughput Tests
# =============================================================================

class TestThroughput:
    """Test throughput meets minimum requirements."""

    @pytest.mark.slow
    @pytest.mark.requires_model
    def test_minimum_throughput(self, model_bundle, videotest3_crop_paths):
        """Test minimum throughput of 2.5 images/second."""
        if model_bundle is None:
            pytest.skip("Model not available")

        images = [load_crop_image(p) for p in videotest3_crop_paths]

        elapsed = measure_inference_time(model_bundle, images, warmup=True)
        throughput = len(images) / elapsed

        print(f"\nThroughput: {throughput:.1f} images/sec")
        print(f"Minimum required: {MIN_THROUGHPUT_IMAGES_PER_SEC} images/sec")

        assert throughput >= MIN_THROUGHPUT_IMAGES_PER_SEC, (
            f"Throughput too low: {throughput:.1f} < {MIN_THROUGHPUT_IMAGES_PER_SEC} images/sec"
        )

    @pytest.mark.slow
    @pytest.mark.requires_model
    def test_batch_vs_sequential(self, model_bundle, videotest3_crop_paths):
        """
        Compare batch vs sequential processing.

        Batch processing should be significantly faster.
        """
        if model_bundle is None:
            pytest.skip("Model not available")

        # Use subset for faster test
        images = [load_crop_image(p) for p in videotest3_crop_paths[:8]]

        # Warmup
        _ = model_bundle.recognize_batch([images[0]])

        # Batch processing
        batch_start = time.perf_counter()
        _ = model_bundle.recognize_batch(images)
        batch_time = time.perf_counter() - batch_start

        # Sequential processing
        seq_start = time.perf_counter()
        for img in images:
            _ = model_bundle.recognize_batch([img])
        seq_time = time.perf_counter() - seq_start

        speedup = seq_time / batch_time

        print(f"\n{'='*60}")
        print(f"BATCH vs SEQUENTIAL ({len(images)} images)")
        print(f"{'='*60}")
        print(f"Batch time: {batch_time:.2f}s")
        print(f"Sequential time: {seq_time:.2f}s")
        print(f"Speedup: {speedup:.1f}x")

        # Batch should be at least as fast as sequential (on CPU, speedup is modest)
        assert speedup >= 1.0, (
            f"Batch processing slower than sequential: {speedup:.1f}x"
        )


# =============================================================================
# Performance Regression Tests
# =============================================================================

class TestPerformanceRegression:
    """Detect performance regressions between versions."""

    @pytest.mark.slow
    @pytest.mark.requires_model
    def test_no_performance_regression(self, model_bundle, videotest3_crop_paths):
        """
        Ensure performance hasn't regressed from baseline.

        Current baseline: 3.79s for 16 images (with 10% buffer: 4.2s)
        """
        if model_bundle is None:
            pytest.skip("Model not available")

        images = [load_crop_image(p) for p in videotest3_crop_paths]

        # Run multiple times and take median
        times = []
        for _ in range(3):
            elapsed = measure_inference_time(model_bundle, images, warmup=True)
            times.append(elapsed)

        median_time = sorted(times)[len(times)//2]

        # Allow 10% regression from baseline
        max_allowed = CURRENT_BASELINE_16_IMAGES * 1.1

        print(f"\n{'='*60}")
        print(f"REGRESSION CHECK")
        print(f"{'='*60}")
        print(f"Median time: {median_time:.2f}s")
        print(f"Baseline: {CURRENT_BASELINE_16_IMAGES}s")
        print(f"Max allowed (baseline + 10%): {max_allowed:.2f}s")
        print(f"All runs: {[f'{t:.2f}s' for t in times]}")

        assert median_time < max_allowed, (
            f"PERFORMANCE REGRESSION: {median_time:.2f}s > {max_allowed:.2f}s (baseline + 10%)"
        )


# =============================================================================
# Model Loading Tests
# =============================================================================

class TestModelLoading:
    """Test model loading performance."""

    def test_model_load_time_recorded(self, model_bundle):
        """Verify model load time was recorded."""
        if model_bundle is None:
            pytest.skip("Model not available")

        assert hasattr(model_bundle, 'load_time_seconds')
        print(f"\nModel load time: {model_bundle.load_time_seconds:.1f}s")

    def test_model_device(self, model_bundle):
        """Verify model is on expected device."""
        if model_bundle is None:
            pytest.skip("Model not available")

        print(f"\nModel device: {model_bundle.device}")

        # MPS or CUDA preferred for performance
        if model_bundle.device == "cpu":
            print("[WARN] Running on CPU - performance may be suboptimal")


# =============================================================================
# Memory Tests
# =============================================================================

class TestMemoryUsage:
    """Test memory usage during inference."""

    @pytest.mark.slow
    @pytest.mark.requires_model
    def test_no_memory_leak(self, model_bundle, videotest3_crop_paths):
        """
        Test that repeated inference doesn't leak memory.

        Run inference multiple times and check memory growth is bounded.
        """
        if model_bundle is None:
            pytest.skip("Model not available")

        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            pytest.skip("psutil not installed for memory monitoring")

        images = [load_crop_image(p) for p in videotest3_crop_paths[:8]]

        # Initial memory
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # Run inference multiple times
        for _ in range(5):
            _ = model_bundle.recognize_batch(images)

        # Final memory
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        growth = final_memory - initial_memory

        print(f"\n{'='*60}")
        print(f"MEMORY USAGE")
        print(f"{'='*60}")
        print(f"Initial: {initial_memory:.0f} MB")
        print(f"Final: {final_memory:.0f} MB")
        print(f"Growth: {growth:.0f} MB")

        # Allow up to 500MB growth (model caching, etc.)
        assert growth < 500, (
            f"Possible memory leak: {growth:.0f}MB growth after 5 inference runs"
        )


# =============================================================================
# Benchmark Suite (for detailed profiling)
# =============================================================================

@pytest.mark.benchmark
class TestBenchmarks:
    """Detailed benchmark tests for profiling."""

    @pytest.mark.slow
    @pytest.mark.requires_model
    def test_full_benchmark(self, model_bundle, videotest3_crop_paths, videotest2_crop_paths):
        """
        Run comprehensive benchmark on all test images.

        Reports detailed timing breakdown.
        """
        if model_bundle is None:
            pytest.skip("Model not available")

        results = []

        # Benchmark videotest3
        images3 = [load_crop_image(p) for p in videotest3_crop_paths]
        _ = model_bundle.recognize_batch([images3[0]])  # warmup

        times3 = []
        for _ in range(3):
            t = measure_inference_time(model_bundle, images3, warmup=False)
            times3.append(t)

        results.append({
            "dataset": "videotest3",
            "images": len(images3),
            "min_time": min(times3),
            "max_time": max(times3),
            "avg_time": sum(times3)/len(times3),
        })

        # Benchmark videotest2
        images2 = [load_crop_image(p) for p in videotest2_crop_paths]

        times2 = []
        for _ in range(3):
            t = measure_inference_time(model_bundle, images2, warmup=False)
            times2.append(t)

        results.append({
            "dataset": "videotest2",
            "images": len(images2),
            "min_time": min(times2),
            "max_time": max(times2),
            "avg_time": sum(times2)/len(times2),
        })

        # Print report
        print(f"\n{'='*70}")
        print(f"BENCHMARK RESULTS")
        print(f"{'='*70}")
        print(f"Device: {model_bundle.device}")
        print(f"Model load time: {model_bundle.load_time_seconds:.1f}s")
        print()

        for r in results:
            print(f"{r['dataset']}:")
            print(f"  Images: {r['images']}")
            print(f"  Min time: {r['min_time']:.2f}s")
            print(f"  Max time: {r['max_time']:.2f}s")
            print(f"  Avg time: {r['avg_time']:.2f}s")
            print(f"  Throughput: {r['images']/r['avg_time']:.1f} img/s")
            print()
