#!/usr/bin/env python3
"""
GNHK Dataset Preprocessor - Crop Word/Line Regions

The GNHK dataset contains full-page scans with word-level polygon annotations.
TrOCR expects single-line images. This script:

1. Reads the manifest with word-level bounding boxes
2. Groups words into lines based on Y-coordinate proximity
3. Crops line regions from the full-page images
4. Creates a new dataset suitable for TrOCR training

Usage:
    python preprocess_gnhk.py --input ../datasets/gnhk --output ../datasets/gnhk_lines
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from PIL import Image
import numpy as np


@dataclass
class WordAnnotation:
    """Single word annotation from GNHK manifest."""
    text: str
    polygon: List[Dict[str, int]]

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Get axis-aligned bounding box (x_min, y_min, x_max, y_max)."""
        xs = [p['x'] for p in self.polygon]
        ys = [p['y'] for p in self.polygon]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def center_y(self) -> float:
        """Get Y-center for line grouping."""
        _, y_min, _, y_max = self.bbox
        return (y_min + y_max) / 2

    @property
    def center_x(self) -> float:
        """Get X-center for word ordering."""
        x_min, _, x_max, _ = self.bbox
        return (x_min + x_max) / 2


@dataclass
class LineRegion:
    """A group of words forming a line."""
    words: List[WordAnnotation]

    @property
    def text(self) -> str:
        """Get concatenated text of all words."""
        # Sort words left-to-right
        sorted_words = sorted(self.words, key=lambda w: w.center_x)
        return ' '.join(w.text for w in sorted_words)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Get bounding box encompassing all words."""
        all_bboxes = [w.bbox for w in self.words]
        x_min = min(b[0] for b in all_bboxes)
        y_min = min(b[1] for b in all_bboxes)
        x_max = max(b[2] for b in all_bboxes)
        y_max = max(b[3] for b in all_bboxes)
        return (x_min, y_min, x_max, y_max)


def load_manifest(manifest_path: Path) -> List[Dict]:
    """Load GNHK SageMaker manifest file."""
    records = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    return records


def parse_annotations(record: Dict) -> Tuple[str, List[WordAnnotation]]:
    """Parse word annotations from a manifest record."""
    image_path = record.get('source-ref', '')
    annotations = record.get('annotations', {})
    texts = annotations.get('texts', [])

    words = []
    for text_ann in texts:
        text = text_ann.get('text', '')
        polygon = text_ann.get('polygon', [])

        # Skip empty or placeholder text
        if not text or '%' in text:  # Skip %SC%, %NA%, etc.
            continue

        # Skip if polygon is invalid
        if len(polygon) < 3:
            continue

        words.append(WordAnnotation(text=text, polygon=polygon))

    return image_path, words


def group_words_into_lines(words: List[WordAnnotation],
                           y_threshold: float = 50.0) -> List[LineRegion]:
    """
    Group words into lines based on Y-coordinate proximity.

    Words with similar Y-centers are considered to be on the same line.
    """
    if not words:
        return []

    # Sort words by Y-center
    sorted_words = sorted(words, key=lambda w: w.center_y)

    lines = []
    current_line = [sorted_words[0]]
    current_y = sorted_words[0].center_y

    for word in sorted_words[1:]:
        if abs(word.center_y - current_y) <= y_threshold:
            # Same line
            current_line.append(word)
        else:
            # New line
            lines.append(LineRegion(words=current_line))
            current_line = [word]
            current_y = word.center_y

    # Don't forget the last line
    if current_line:
        lines.append(LineRegion(words=current_line))

    return lines


def crop_region(image: Image.Image, bbox: Tuple[int, int, int, int],
                padding: int = 10) -> Optional[Image.Image]:
    """
    Crop a region from an image with padding.

    Args:
        image: Source PIL image
        bbox: (x_min, y_min, x_max, y_max) bounding box
        padding: Pixels of padding to add around the crop

    Returns:
        Cropped PIL image, or None if bbox is invalid
    """
    x_min, y_min, x_max, y_max = bbox

    # Validate bounding box
    if x_min >= x_max or y_min >= y_max:
        return None

    # Add padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image.width, x_max + padding)
    y_max = min(image.height, y_max + padding)

    # Final validation after padding adjustments
    if x_min >= x_max or y_min >= y_max:
        return None

    return image.crop((x_min, y_min, x_max, y_max))


def process_image(image_path: Path, words: List[WordAnnotation],
                  output_dir: Path,
                  mode: str = 'line',
                  y_threshold: float = 50.0,
                  min_words_per_line: int = 1,
                  max_text_length: int = 100) -> List[Dict]:
    """
    Process a single image and create word/line crops.

    Args:
        image_path: Path to source image
        words: List of word annotations
        output_dir: Directory for output crops
        mode: 'word' for individual words, 'line' for grouped lines
        y_threshold: Y-distance threshold for line grouping
        min_words_per_line: Minimum words to include a line
        max_text_length: Maximum text length to include

    Returns:
        List of sample dicts with 'image_path' and 'text'
    """
    samples = []

    if not image_path.exists():
        print(f"Warning: Image not found: {image_path}")
        return samples

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return samples

    base_name = image_path.stem

    if mode == 'word':
        # Create individual word crops
        for i, word in enumerate(words):
            if len(word.text) > max_text_length:
                continue

            crop = crop_region(image, word.bbox, padding=5)

            # Skip invalid crops
            if crop is None:
                continue

            # Skip very small crops
            if crop.width < 10 or crop.height < 10:
                continue

            output_path = output_dir / f"{base_name}_w{i:04d}.png"
            crop.save(output_path)

            samples.append({
                'image_path': str(output_path),
                'text': word.text
            })

    elif mode == 'line':
        # Group into lines and create line crops
        lines = group_words_into_lines(words, y_threshold)

        for i, line in enumerate(lines):
            if len(line.words) < min_words_per_line:
                continue

            text = line.text
            if len(text) > max_text_length:
                continue

            # Skip lines with only punctuation or very short text
            if len(text.strip()) < 2:
                continue

            crop = crop_region(image, line.bbox, padding=15)

            # Skip invalid crops
            if crop is None:
                continue

            # Skip very small crops
            if crop.width < 20 or crop.height < 10:
                continue

            output_path = output_dir / f"{base_name}_l{i:04d}.png"
            crop.save(output_path)

            samples.append({
                'image_path': str(output_path),
                'text': text
            })

    return samples


def create_manifest(samples: List[Dict], output_path: Path):
    """Create a SageMaker-style manifest file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            # Use relative path in manifest
            image_name = Path(sample['image_path']).name
            record = {
                'source-ref': image_name,
                'annotations': {
                    'texts': [{'text': sample['text'], 'polygon': []}]
                }
            }
            f.write(json.dumps(record) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess GNHK dataset for TrOCR training'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input GNHK dataset directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for processed dataset')
    parser.add_argument('--mode', type=str, choices=['word', 'line'],
                        default='line',
                        help='Crop mode: word or line level')
    parser.add_argument('--y-threshold', type=float, default=50.0,
                        help='Y-distance threshold for line grouping')
    parser.add_argument('--min-words', type=int, default=2,
                        help='Minimum words per line')
    parser.add_argument('--max-text-length', type=int, default=100,
                        help='Maximum text length to include')
    parser.add_argument('--split', type=str, choices=['train', 'test', 'both'],
                        default='both',
                        help='Which split to process')

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    splits_to_process = ['train', 'test'] if args.split == 'both' else [args.split]

    for split in splits_to_process:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")

        split_input = input_dir / split
        manifest_path = split_input / f"{split}.manifest"

        if not manifest_path.exists():
            print(f"Manifest not found: {manifest_path}")
            continue

        split_output = output_dir / split
        split_output.mkdir(parents=True, exist_ok=True)

        # Load manifest
        records = load_manifest(manifest_path)
        print(f"Loaded {len(records)} records from manifest")

        all_samples = []

        for record in records:
            image_name, words = parse_annotations(record)

            if not words:
                continue

            image_path = split_input / image_name

            samples = process_image(
                image_path=image_path,
                words=words,
                output_dir=split_output,
                mode=args.mode,
                y_threshold=args.y_threshold,
                min_words_per_line=args.min_words,
                max_text_length=args.max_text_length
            )

            all_samples.extend(samples)

        # Create output manifest
        output_manifest = split_output / f"{split}.manifest"
        create_manifest(all_samples, output_manifest)

        print(f"\nCreated {len(all_samples)} samples")
        print(f"Output manifest: {output_manifest}")

        # Print sample statistics
        if all_samples:
            texts = [s['text'] for s in all_samples]
            avg_len = sum(len(t) for t in texts) / len(texts)
            max_len = max(len(t) for t in texts)
            print(f"Average text length: {avg_len:.1f} characters")
            print(f"Max text length: {max_len} characters")

    print(f"\n{'='*60}")
    print("Preprocessing complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
