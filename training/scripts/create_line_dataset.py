#!/usr/bin/env python3
"""
Create Line-Level Training Data from GNHK Word Annotations

GNHK provides word-level polygons. This script:
1. Groups words into lines based on Y-coordinate clustering
2. Crops line images from full pages
3. Creates ground truth text files for each line

This produces the training data format TrOCR expects:
  - image_001_line_01.jpg + image_001_line_01.txt
"""

import json
import os
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Any


def load_manifest(manifest_path: str) -> List[Dict]:
    """Load GNHK manifest file (JSONL format)."""
    records = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return records


def get_word_center_y(word: Dict) -> float:
    """Get the vertical center of a word's bounding polygon."""
    polygon = word.get('polygon', [])
    if not polygon:
        return 0
    y_coords = [p.get('y', 0) for p in polygon]
    return sum(y_coords) / len(y_coords) if y_coords else 0


def get_word_bbox(word: Dict) -> Tuple[int, int, int, int]:
    """Get bounding box (x1, y1, x2, y2) from polygon."""
    polygon = word.get('polygon', [])
    if not polygon:
        return (0, 0, 0, 0)

    x_coords = [p.get('x', 0) for p in polygon]
    y_coords = [p.get('y', 0) for p in polygon]

    return (
        min(x_coords),
        min(y_coords),
        max(x_coords),
        max(y_coords)
    )


def cluster_words_into_lines(words: List[Dict], line_height_threshold: float = 50) -> List[List[Dict]]:
    """
    Cluster words into lines based on Y-coordinate proximity.

    Args:
        words: List of word dictionaries with polygon coordinates
        line_height_threshold: Maximum Y-distance to consider words on same line

    Returns:
        List of lines, each line is a list of words sorted by X coordinate
    """
    if not words:
        return []

    # Sort words by vertical center
    words_with_y = [(word, get_word_center_y(word)) for word in words]
    words_with_y.sort(key=lambda x: x[1])

    lines = []
    current_line = [words_with_y[0][0]]
    current_y = words_with_y[0][1]

    for word, y in words_with_y[1:]:
        if abs(y - current_y) <= line_height_threshold:
            # Same line
            current_line.append(word)
        else:
            # New line
            # Sort current line by X coordinate (left to right)
            current_line.sort(key=lambda w: get_word_bbox(w)[0])
            lines.append(current_line)
            current_line = [word]
            current_y = y

    # Don't forget the last line
    if current_line:
        current_line.sort(key=lambda w: get_word_bbox(w)[0])
        lines.append(current_line)

    return lines


def get_line_bbox(words: List[Dict], padding: int = 10) -> Tuple[int, int, int, int]:
    """Get bounding box for entire line with padding."""
    all_x1, all_y1, all_x2, all_y2 = [], [], [], []

    for word in words:
        x1, y1, x2, y2 = get_word_bbox(word)
        all_x1.append(x1)
        all_y1.append(y1)
        all_x2.append(x2)
        all_y2.append(y2)

    return (
        max(0, min(all_x1) - padding),
        max(0, min(all_y1) - padding),
        max(all_x2) + padding,
        max(all_y2) + padding
    )


def get_line_text(words: List[Dict]) -> str:
    """Concatenate word texts to form line text."""
    texts = [word.get('text', '') for word in words]
    return ' '.join(texts).strip()


def process_manifest(
    manifest_path: str,
    image_dir: str,
    output_dir: str,
    line_height_threshold: float = 50,
    min_line_width: int = 50,
    min_words_per_line: int = 1
):
    """
    Process GNHK manifest and create line-level dataset.

    Args:
        manifest_path: Path to GNHK manifest file
        image_dir: Directory containing source images
        output_dir: Output directory for line images and text files
        line_height_threshold: Y-distance threshold for line clustering
        min_line_width: Minimum line width to include
        min_words_per_line: Minimum words per line
    """
    os.makedirs(output_dir, exist_ok=True)

    records = load_manifest(manifest_path)
    print(f"Loaded {len(records)} images from manifest")

    total_lines = 0
    skipped_images = 0

    # Create manifest for the line dataset
    line_manifest = []

    for record in records:
        image_name = record.get('source-ref', '')
        annotations = record.get('annotations', {})
        words = annotations.get('texts', [])

        if not words:
            continue

        # Find image
        image_path = Path(image_dir) / image_name
        if not image_path.exists():
            skipped_images += 1
            continue

        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            skipped_images += 1
            continue

        # Cluster words into lines
        lines = cluster_words_into_lines(words, line_height_threshold)

        # Extract each line
        base_name = Path(image_name).stem

        for line_idx, line_words in enumerate(lines):
            if len(line_words) < min_words_per_line:
                continue

            # Get line bounding box
            x1, y1, x2, y2 = get_line_bbox(line_words)

            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.width, x2)
            y2 = min(img.height, y2)

            line_width = x2 - x1
            line_height = y2 - y1

            if line_width < min_line_width or line_height < 10:
                continue

            # Crop line image
            try:
                line_img = img.crop((x1, y1, x2, y2))
            except Exception as e:
                print(f"Error cropping line from {image_name}: {e}")
                continue

            # Get line text
            line_text = get_line_text(line_words)

            if not line_text.strip():
                continue

            # Save line image and text
            line_name = f"{base_name}_line_{line_idx:03d}"
            line_img_path = Path(output_dir) / f"{line_name}.jpg"
            line_txt_path = Path(output_dir) / f"{line_name}.txt"

            line_img.save(line_img_path, quality=95)
            with open(line_txt_path, 'w', encoding='utf-8') as f:
                f.write(line_text)

            line_manifest.append({
                'image': f"{line_name}.jpg",
                'text': line_text,
                'source_image': image_name,
                'words': len(line_words)
            })

            total_lines += 1

    # Save manifest
    manifest_output = Path(output_dir) / "manifest.json"
    with open(manifest_output, 'w', encoding='utf-8') as f:
        json.dump(line_manifest, f, indent=2)

    print(f"\nCreated {total_lines} line samples")
    print(f"Skipped {skipped_images} images (not found)")
    print(f"Output directory: {output_dir}")
    print(f"Manifest: {manifest_output}")

    return total_lines


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create line-level dataset from GNHK")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to GNHK manifest file")
    parser.add_argument("--images", type=str, required=True,
                        help="Directory containing source images")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for line dataset")
    parser.add_argument("--line-threshold", type=float, default=50,
                        help="Y-distance threshold for line clustering (default: 50)")
    parser.add_argument("--min-width", type=int, default=50,
                        help="Minimum line width in pixels (default: 50)")
    parser.add_argument("--min-words", type=int, default=1,
                        help="Minimum words per line (default: 1)")

    args = parser.parse_args()

    process_manifest(
        manifest_path=args.manifest,
        image_dir=args.images,
        output_dir=args.output,
        line_height_threshold=args.line_threshold,
        min_line_width=args.min_width,
        min_words_per_line=args.min_words
    )


if __name__ == "__main__":
    main()
