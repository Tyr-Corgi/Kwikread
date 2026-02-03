#!/usr/bin/env python3
"""
Add videotest3 data to training dataset and retrain model v3.

This script:
1. Loads videotest3 crops and ground truth labels
2. Creates augmented versions (brightness, contrast, rotation)
3. Adds to existing merged dataset
4. Retrains model to create v3
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance

# Paths
BASE_DIR = Path(__file__).parent.parent
FINETUNE_DIR = BASE_DIR / "finetune"
CROPS_DIR = FINETUNE_DIR / "crops" / "videotest3"
DATASET_DIR = FINETUNE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
GT_PATH = FINETUNE_DIR / "ground_truth.json"


def augment_image(image: Image.Image, prefix: str) -> dict:
    """Create augmented versions of an image."""
    augmented = {}

    # Original
    augmented[f"{prefix}.png"] = image.copy()

    # Brightness variations
    for factor in [0.8, 1.2]:
        enhancer = ImageEnhance.Brightness(image)
        aug = enhancer.enhance(factor)
        augmented[f"{prefix}_bright{factor}.png"] = aug

    # Contrast variations
    for factor in [0.8, 1.2]:
        enhancer = ImageEnhance.Contrast(image)
        aug = enhancer.enhance(factor)
        augmented[f"{prefix}_contrast{factor}.png"] = aug

    # Rotation variations
    for angle in [-3, 3]:
        aug = image.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        augmented[f"{prefix}_rot{angle}.png"] = aug

    return augmented


def add_videotest3_data():
    """Add videotest3 crops to training dataset with augmentation."""

    # Load ground truth
    with open(GT_PATH) as f:
        gt = json.load(f)

    videotest3 = gt.get("videotest3", {})
    labels = videotest3.get("labels", {})

    if not labels or "_note" in labels:
        print("No videotest3 labels found!")
        return 0

    # Count existing samples
    train_metadata = TRAIN_DIR / "metadata.jsonl"
    existing_count = 0
    if train_metadata.exists():
        with open(train_metadata) as f:
            existing_count = sum(1 for _ in f)

    print(f"Existing training samples: {existing_count}")
    print(f"Adding videotest3 samples: {len(labels)} base + augmentations")

    # Split: 12 for train, 4 for val
    items = list(labels.items())
    train_items = items[:12]
    val_items = items[12:]

    new_train = 0
    new_val = 0

    # Add training samples
    train_entries = []
    train_idx = existing_count  # Start from existing count

    for filename, text in train_items:
        crop_path = CROPS_DIR / filename
        if not crop_path.exists():
            print(f"Warning: {crop_path} not found")
            continue

        image = Image.open(crop_path).convert("RGB")
        augmented = augment_image(image, f"train_{train_idx:04d}")

        for aug_filename, aug_image in augmented.items():
            out_path = TRAIN_DIR / aug_filename
            aug_image.save(out_path)
            train_entries.append({"file_name": aug_filename, "text": text})
            new_train += 1

        train_idx += 1

    # Append to train metadata
    with open(train_metadata, "a") as f:
        for entry in train_entries:
            f.write(json.dumps(entry) + "\n")

    # Add validation samples
    val_metadata = VAL_DIR / "metadata.jsonl"
    val_entries = []

    # Count existing val samples
    existing_val = 0
    if val_metadata.exists():
        with open(val_metadata) as f:
            existing_val = sum(1 for _ in f)

    val_idx = existing_val

    for filename, text in val_items:
        crop_path = CROPS_DIR / filename
        if not crop_path.exists():
            print(f"Warning: {crop_path} not found")
            continue

        image = Image.open(crop_path).convert("RGB")
        augmented = augment_image(image, f"val_{val_idx:04d}")

        for aug_filename, aug_image in augmented.items():
            out_path = VAL_DIR / aug_filename
            aug_image.save(out_path)
            val_entries.append({"file_name": aug_filename, "text": text})
            new_val += 1

        val_idx += 1

    # Append to val metadata
    with open(val_metadata, "a") as f:
        for entry in val_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nAdded {new_train} training samples")
    print(f"Added {new_val} validation samples")

    # Count total
    with open(train_metadata) as f:
        total_train = sum(1 for _ in f)
    with open(val_metadata) as f:
        total_val = sum(1 for _ in f)

    print(f"\nTotal training samples: {total_train}")
    print(f"Total validation samples: {total_val}")

    return new_train + new_val


def train_model_v3():
    """Train model v3 on updated dataset."""
    import torch
    from transformers import (
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        default_data_collator,
    )
    from transformers.trainer_callback import EarlyStoppingCallback
    from torch.utils.data import Dataset

    class GroceryDataset(Dataset):
        def __init__(self, data_dir, processor, max_length=64):
            self.data_dir = Path(data_dir)
            self.processor = processor
            self.max_length = max_length
            self.samples = []

            metadata_path = self.data_dir / "metadata.jsonl"
            with open(metadata_path) as f:
                for line in f:
                    item = json.loads(line)
                    img_path = self.data_dir / item["file_name"]
                    if img_path.exists():
                        self.samples.append({
                            "image_path": str(img_path),
                            "text": item["text"]
                        })

            print(f"Loaded {len(self.samples)} samples from {data_dir}")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            image = Image.open(sample["image_path"]).convert("RGB")
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

            labels = self.processor.tokenizer(
                sample["text"],
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.squeeze()

            labels[labels == self.processor.tokenizer.pad_token_id] = -100

            return {"pixel_values": pixel_values, "labels": labels}

    # Use model v2 as base (trained on IAM + grocery)
    model_v2_path = FINETUNE_DIR / "model_v2" / "final"
    if model_v2_path.exists():
        base_model = str(model_v2_path)
        print(f"Using model v2 as base: {base_model}")
    else:
        base_model = "microsoft/trocr-base-handwritten"
        print(f"Model v2 not found, using base: {base_model}")

    output_dir = str(FINETUNE_DIR / "model_v3")

    print(f"Loading model: {base_model}")
    processor = TrOCRProcessor.from_pretrained(base_model)
    model = VisionEncoderDecoderModel.from_pretrained(base_model)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.gradient_checkpointing_enable()

    # Load datasets
    train_dataset = GroceryDataset(str(TRAIN_DIR), processor)
    val_dataset = GroceryDataset(str(VAL_DIR), processor)

    # Training arguments - fine-tune on new data
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,  # Fewer epochs since we're fine-tuning v2
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,  # Lower LR for fine-tuning
        warmup_steps=50,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=64,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,
        logging_dir=str(FINETUNE_DIR / "logs_v3"),
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"\nStarting training for model v3...")
    trainer.train()

    # Save final model
    final_dir = Path(output_dir) / "final"
    print(f"\nSaving model v3 to: {final_dir}")
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    print("\n Model v3 training complete!")
    return str(final_dir)


if __name__ == "__main__":
    print("=" * 60)
    print("Adding videotest3 data and training model v3")
    print("=" * 60)

    # Step 1: Add videotest3 data
    print("\n[Step 1] Adding videotest3 data to dataset...")
    added = add_videotest3_data()

    if added > 0:
        # Step 2: Train model v3
        print("\n[Step 2] Training model v3...")
        model_path = train_model_v3()

        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Model v3 saved to: {model_path}")
        print("\nTo use in OCR server, update FINETUNED_MODEL_PATH in ocr_server.py")
    else:
        print("No data added, skipping training")
