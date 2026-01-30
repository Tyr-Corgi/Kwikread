#!/usr/bin/env python3
"""
Generate pseudo-labels for unlabeled images using the current fine-tuned model.
This enables self-training to continue fine-tuning without new labeled data.
"""

import os
import json
from pathlib import Path
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor


def generate_pseudo_labels(
    model_path: str,
    image_dir: str,
    output_file: str,
    device: str = "mps"
):
    """Generate pseudo-labels for images using the fine-tuned model."""

    print(f"Loading model from: {model_path}")
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    processor = TrOCRProcessor.from_pretrained(model_path)

    model = model.to(device)
    model.eval()

    # Get all images
    image_files = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    print(f"Found {len(image_files)} images")

    results = []

    with torch.no_grad():
        for i, img_path in enumerate(image_files):
            try:
                # Load and process image
                image = Image.open(img_path).convert("RGB")
                pixel_values = processor(images=image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(device)

                # Generate text
                generated_ids = model.generate(pixel_values, max_length=64)
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Skip empty or very short predictions
                if len(text.strip()) < 2:
                    continue

                results.append({
                    "image": str(img_path.name),
                    "text": text.strip()
                })

                if (i + 1) % 20 == 0:
                    print(f"Processed {i + 1}/{len(image_files)}: {text[:50]}...")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    print(f"\nGenerated {len(results)} pseudo-labels")

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {output_file}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./checkpoints/best")
    parser.add_argument("--images", default="./datasets/gnhk/test")
    parser.add_argument("--output", default="./data/pseudo_labels.json")
    parser.add_argument("--device", default="mps")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_pseudo_labels(args.model, args.images, args.output, args.device)
