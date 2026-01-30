#!/usr/bin/env python3
"""
GNHK Dataset Loader for TrOCR Fine-tuning

Handles loading and preprocessing of the GNHK handwriting dataset
for training TrOCR models.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import TrOCRProcessor
import numpy as np


class GNHKDataset(Dataset):
    """
    PyTorch Dataset for GNHK Handwriting data.

    Supports:
    - Line-level dataset (image.jpg + image.txt pairs)
    - SageMaker manifest format
    - JSON annotations
    """

    def __init__(
        self,
        data_dir: str,
        processor: TrOCRProcessor,
        split: str = "train",
        max_length: int = 128,
        image_size: Tuple[int, int] = (384, 384),
        augment: bool = False,
        augment_strength: str = 'medium'
    ):
        """
        Initialize the GNHK dataset.

        Args:
            data_dir: Path to the GNHK dataset directory
            processor: TrOCR processor for tokenization
            split: 'train' or 'test'
            max_length: Maximum token sequence length
            image_size: Target image size (width, height)
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.split = split
        self.max_length = max_length
        self.image_size = image_size
        self.augment = augment
        self.augment_strength = augment_strength

        # Initialize advanced augmentor if augmentation is enabled
        self.augmentor = None
        if augment:
            try:
                from augmentation import HandwritingAugmentor
                self.augmentor = HandwritingAugmentor(strength=augment_strength)
                print(f"Using advanced augmentation (strength={augment_strength})")
            except ImportError:
                print("Advanced augmentation not available, using basic augmentation")

        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load samples from GNHK dataset."""
        samples = []

        # Try line-level manifest format first (our new format)
        line_manifest_paths = [
            self.data_dir / self.split / "manifest.json",
            self.data_dir / "manifest.json",
        ]
        for manifest_path in line_manifest_paths:
            if manifest_path.exists():
                samples = self._load_line_manifest(manifest_path)
                if samples:
                    return samples

        # Try SageMaker manifest format
        manifest_paths = [
            self.data_dir / f"{self.split}.manifest",
            self.data_dir / self.split / f"{self.split}.manifest",
        ]
        for manifest_path in manifest_paths:
            if manifest_path.exists():
                samples = self._load_sagemaker_manifest(manifest_path)
                if samples:
                    return samples

        # Try JSON format
        json_patterns = [
            f"{self.split}.json",
            f"{self.split}_annotations.json",
            f"gnhk_{self.split}.json"
        ]
        for pattern in json_patterns:
            json_path = self.data_dir / pattern
            if json_path.exists():
                samples = self._load_json_annotations(json_path)
                if samples:
                    return samples

        # Try to find images with text files
        samples = self._load_from_directory()

        return samples

    def _load_line_manifest(self, manifest_path: Path) -> List[Dict]:
        """Load our line-level manifest format (JSON with image/text pairs)."""
        samples = []
        manifest_dir = manifest_path.parent

        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                image_name = item.get('image', '')
                text = item.get('text', '')

                if not image_name or not text:
                    continue

                image_path = manifest_dir / image_name
                if image_path.exists():
                    samples.append({
                        'image_path': str(image_path),
                        'text': text.strip()
                    })

        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading manifest {manifest_path}: {e}")

        return samples

    def _load_sagemaker_manifest(self, manifest_path: Path) -> List[Dict]:
        """Load SageMaker Ground Truth manifest format."""
        samples = []
        manifest_dir = manifest_path.parent  # Images are in same folder as manifest

        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())

                    # Extract image path
                    image_path = record.get('source-ref', record.get('image', ''))

                    # Handle different annotation key formats
                    text = None

                    # GNHK format: annotations.texts[].text (word-level polygons)
                    if 'annotations' in record and 'texts' in record['annotations']:
                        words = record['annotations']['texts']
                        if words:
                            # Concatenate all words to form full text
                            text = ' '.join(w.get('text', '') for w in words if w.get('text'))

                    # Fallback to other formats
                    if not text:
                        for key in ['text', 'label', 'transcription', 'gnhk-label']:
                            if key in record:
                                if isinstance(record[key], dict):
                                    text = record[key].get('text', record[key].get('label'))
                                else:
                                    text = record[key]
                                break

                    if image_path and text:
                        # Convert S3 path to local path if needed
                        if image_path.startswith('s3://'):
                            image_path = image_path.split('/')[-1]

                        # Build full image path (relative to manifest location)
                        full_image_path = manifest_dir / image_path
                        if not full_image_path.exists():
                            full_image_path = self.data_dir / image_path

                        if full_image_path.exists():
                            samples.append({
                                'image_path': str(full_image_path),
                                'text': str(text).strip()
                            })

                except json.JSONDecodeError:
                    continue

        return samples

    def _load_json_annotations(self, json_path: Path) -> List[Dict]:
        """Load Paper JSON format annotations."""
        samples = []

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get('annotations', data.get('images', data.get('data', [])))
        else:
            return samples

        for item in items:
            image_path = item.get('image', item.get('file_name', item.get('image_path', '')))
            text = item.get('text', item.get('transcription', item.get('label', '')))

            if image_path and text:
                full_path = self.data_dir / image_path
                if not full_path.exists():
                    full_path = self.data_dir / "images" / image_path

                samples.append({
                    'image_path': str(full_path),
                    'text': str(text).strip()
                })

        return samples

    def _load_from_directory(self) -> List[Dict]:
        """Load samples by scanning directory for image-text pairs."""
        samples = []

        # Find all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []

        for ext in image_extensions:
            image_files.extend(self.data_dir.rglob(ext))

        for img_path in image_files:
            # Look for corresponding text file
            txt_path = img_path.with_suffix('.txt')
            if txt_path.exists():
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                if text:
                    samples.append({
                        'image_path': str(img_path),
                        'text': text
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.samples[idx]

        # Load image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', self.image_size, color='white')

        # Apply augmentation if enabled
        if self.augment:
            image = self._augment_image(image)

        # Process image
        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Process text
        labels = self.processor.tokenizer(
            sample['text'],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        # Replace padding token id with -100 for loss calculation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

    def _augment_image(self, image: Image.Image) -> Image.Image:
        """Apply data augmentation using advanced augmentor if available."""
        import random

        # Use advanced augmentor if available
        if self.augmentor is not None:
            return self.augmentor(image)

        # Fallback to basic augmentation
        # Random rotation (-5 to 5 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)
            image = image.rotate(angle, fillcolor='white')

        # Random brightness adjustment
        if random.random() > 0.5:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)

        # Random contrast adjustment
        if random.random() > 0.5:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)

        return image


class CorrectionDataset(Dataset):
    """
    Dataset for loading corrections from kwikread's CorrectionTracker.

    Uses human corrections to fine-tune the model on domain-specific data.
    """

    def __init__(
        self,
        corrections_file: str,
        processor: TrOCRProcessor,
        max_length: int = 128
    ):
        """
        Initialize from corrections JSON file.

        Args:
            corrections_file: Path to corrections.json from kwikread
            processor: TrOCR processor
            max_length: Maximum token sequence length
        """
        self.processor = processor
        self.max_length = max_length

        self.samples = self._load_corrections(corrections_file)
        print(f"Loaded {len(self.samples)} correction samples")

    def _load_corrections(self, corrections_file: str) -> List[Dict]:
        """Load corrections from kwikread format."""
        samples = []

        if not os.path.exists(corrections_file):
            print(f"Corrections file not found: {corrections_file}")
            return samples

        with open(corrections_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        records = data.get('Records', data.get('records', []))

        for record in records:
            # Only use human rejections (corrections)
            event_type = record.get('EventType', record.get('event_type', ''))
            if event_type == 'human_rejection':
                image_path = record.get('ImagePath', record.get('image_path', ''))
                corrected_text = record.get('CorrectedText', record.get('corrected_text', ''))

                if image_path and corrected_text and os.path.exists(image_path):
                    samples.append({
                        'image_path': image_path,
                        'text': corrected_text
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.samples[idx]

        # Load and process image
        image = Image.open(sample['image_path']).convert('RGB')

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Process text
        labels = self.processor.tokenizer(
            sample['text'],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


def create_dataloaders(
    data_dir: str,
    processor: TrOCRProcessor,
    batch_size: int = 8,
    eval_batch_size: int = None,
    num_workers: int = 4,
    augment_train: bool = True,
    augment_strength: str = 'medium'
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Path to dataset directory
        processor: TrOCR processor
        batch_size: Batch size for training
        eval_batch_size: Batch size for validation (defaults to batch_size)
        num_workers: Number of data loading workers
        augment_train: Whether to augment training data
        augment_strength: Augmentation strength ('light', 'medium', 'heavy')

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    if eval_batch_size is None:
        eval_batch_size = batch_size
    train_dataset = GNHKDataset(
        data_dir=data_dir,
        processor=processor,
        split="train",
        augment=augment_train,
        augment_strength=augment_strength
    )

    val_dataset = GNHKDataset(
        data_dir=data_dir,
        processor=processor,
        split="test",
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    from transformers import TrOCRProcessor

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    # Test with sample data
    dataset = GNHKDataset(
        data_dir="./datasets/gnhk",
        processor=processor,
        split="train"
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample pixel_values shape: {sample['pixel_values'].shape}")
        print(f"Sample labels shape: {sample['labels'].shape}")
