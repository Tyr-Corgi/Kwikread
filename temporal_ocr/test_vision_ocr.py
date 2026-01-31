#!/usr/bin/env python3
"""
Test vision LLM OCR against TrOCR on actual crop images.
This script tests if a local vision model can read handwritten text better than TrOCR.
"""

import os
import sys
import json
import requests
import base64
from pathlib import Path

# Test crops and their ground truth (from user feedback)
TEST_CASES = [
    ("line_13.png", "Oat Milk", "out milk"),      # TrOCR output: "out milk"
    ("line_14.png", "Protein Bars", "problem Bars"),  # TrOCR output: "problem Bars"
    ("line_09.png", "Apple Sauce", "Apple since"),    # TrOCR output: "Apple since"
    ("line_07.png", "Cheese Sticks", "Cheese"),       # TrOCR output: "Cheese" (truncated)
    ("line_12.png", "Trail Mix", "Trail unit"),       # TrOCR output: "Trail unit"
    ("line_01.png", "Pillsbury Cookie Dough", "philt lbury Cookie Doug"),  # TrOCR output
]

CROPS_DIR = Path("/Users/tygr/Desktop/Repos/kwikread/temporal_ocr/videotest2_v9/crops/videotest2_frame")

def test_ollama_vision(image_path: str, model: str = "moondream") -> str:
    """Test Ollama vision model on an image."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode('utf-8')

    prompt = "Read the handwritten text in this image. Output ONLY the exact text you see, nothing else."

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            return f"[ERROR: {response.status_code}]"
    except Exception as e:
        return f"[ERROR: {e}]"

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two strings."""
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(text1.lower().strip(), text2.lower().strip()) / 100.0
    except ImportError:
        # Simple fallback
        t1, t2 = text1.lower().strip(), text2.lower().strip()
        if t1 == t2:
            return 1.0
        return 0.0

def run_tests(model: str = "moondream"):
    """Run all test cases and report results."""
    print(f"\n{'='*70}")
    print(f"VISION OCR TEST: {model}")
    print(f"{'='*70}\n")

    results = []
    trocr_correct = 0
    vision_correct = 0

    for crop_file, ground_truth, trocr_output in TEST_CASES:
        image_path = CROPS_DIR / crop_file

        if not image_path.exists():
            print(f"[SKIP] {crop_file} not found")
            continue

        # Test vision model
        vision_output = test_ollama_vision(str(image_path), model)

        # Calculate similarities
        trocr_sim = calculate_similarity(trocr_output, ground_truth)
        vision_sim = calculate_similarity(vision_output, ground_truth)

        # Determine which is closer to ground truth
        trocr_win = trocr_sim >= vision_sim

        if trocr_sim > 0.8:
            trocr_correct += 1
        if vision_sim > 0.8:
            vision_correct += 1

        results.append({
            "file": crop_file,
            "ground_truth": ground_truth,
            "trocr": trocr_output,
            "trocr_similarity": trocr_sim,
            "vision": vision_output,
            "vision_similarity": vision_sim,
            "winner": "TrOCR" if trocr_win else "Vision"
        })

        # Print result
        print(f"File: {crop_file}")
        print(f"  Ground Truth: '{ground_truth}'")
        print(f"  TrOCR Output: '{trocr_output}' (sim: {trocr_sim:.0%})")
        print(f"  Vision Output: '{vision_output}' (sim: {vision_sim:.0%})")
        print(f"  Winner: {'TrOCR' if trocr_win else 'VISION ✓'}")
        print()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"TrOCR Correct (>80% similarity): {trocr_correct}/{len(TEST_CASES)}")
    print(f"Vision Correct (>80% similarity): {vision_correct}/{len(TEST_CASES)}")
    print(f"Vision improvement: {vision_correct - trocr_correct:+d} items")

    return results

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "moondream"
    run_tests(model)
