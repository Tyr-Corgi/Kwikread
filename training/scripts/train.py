#!/usr/bin/env python3
"""
TrOCR Fine-tuning Script for Handwriting Recognition

Fine-tunes Microsoft's TrOCR model on the GNHK dataset or custom corrections.

Usage:
    python train.py --config ../configs/train_config.yaml
    python train.py --data_dir ../datasets/gnhk --epochs 10 --batch_size 8
"""

import os
import sys
import argparse
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np

# Local imports
from dataset import GNHKDataset, CorrectionDataset, create_dataloaders
from metrics import calculate_cer, calculate_wer, evaluate_model


class TrOCRTrainer:
    """
    Trainer class for fine-tuning TrOCR on handwriting data.
    """

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        output_dir: str = "./checkpoints",
        device: str = "auto"
    ):
        """
        Initialize the trainer.

        Args:
            model_name: Pretrained model name or path
            output_dir: Directory for saving checkpoints
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load model and processor
        print(f"Loading model: {model_name}")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        # Configure model for training
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        self.model.to(self.device)

        # Training state
        self.global_step = 0
        self.best_cer = float('inf')
        self.training_history = []

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 10,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        save_steps: int = 1000,
        eval_steps: int = 500,
        early_stopping_patience: int = 5,
        use_amp: bool = True,
        log_dir: Optional[str] = None,
        max_eval_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Initial learning rate
            warmup_steps: Number of warmup steps
            gradient_accumulation_steps: Steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            early_stopping_patience: Stop if no improvement for N evaluations
            use_amp: Use automatic mixed precision
            log_dir: Directory for TensorBoard logs

        Returns:
            Dictionary with training results
        """
        # Setup optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Calculate total steps
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps

        # Setup scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Setup AMP scaler
        scaler = GradScaler() if use_amp and self.device.type == "cuda" else None

        # Setup TensorBoard
        writer = None
        if log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(log_dir)
            except ImportError:
                print("TensorBoard not available, skipping logging")

        # Training loop
        print(f"\n{'='*60}")
        print(f"Starting training")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"{'='*60}\n")

        no_improvement_count = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_steps = 0

            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{epochs}",
                leave=True
            )

            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass with AMP
                if use_amp and scaler:
                    with autocast():
                        outputs = self.model(
                            pixel_values=pixel_values,
                            labels=labels
                        )
                        loss = outputs.loss / gradient_accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    outputs = self.model(
                        pixel_values=pixel_values,
                        labels=labels
                    )
                    loss = outputs.loss / gradient_accumulation_steps
                    loss.backward()

                epoch_loss += loss.item() * gradient_accumulation_steps
                epoch_steps += 1

                # Gradient accumulation step
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_grad_norm
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_grad_norm
                        )
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()
                    self.global_step += 1

                    # Update progress bar
                    progress_bar.set_postfix({
                        "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                    })

                    # Log to TensorBoard
                    if writer:
                        writer.add_scalar(
                            "train/loss",
                            loss.item() * gradient_accumulation_steps,
                            self.global_step
                        )
                        writer.add_scalar(
                            "train/lr",
                            scheduler.get_last_lr()[0],
                            self.global_step
                        )

                    # Evaluate
                    if self.global_step % eval_steps == 0:
                        metrics = self._evaluate(val_loader, max_samples=max_eval_samples)

                        if writer:
                            writer.add_scalar("eval/cer", metrics["cer"], self.global_step)
                            writer.add_scalar("eval/wer", metrics["wer"], self.global_step)

                        print(f"\n[Step {self.global_step}] "
                              f"CER: {metrics['cer']:.4f}, WER: {metrics['wer']:.4f}")

                        # Check for improvement
                        if metrics["cer"] < self.best_cer:
                            self.best_cer = metrics["cer"]
                            self._save_checkpoint("best")
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1

                        # Early stopping
                        if no_improvement_count >= early_stopping_patience:
                            print(f"\nEarly stopping after {no_improvement_count} "
                                  f"evaluations without improvement")
                            break

                    # Save checkpoint
                    if self.global_step % save_steps == 0:
                        self._save_checkpoint(f"step_{self.global_step}")

            # End of epoch
            avg_epoch_loss = epoch_loss / epoch_steps
            print(f"\nEpoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")

            self.training_history.append({
                "epoch": epoch + 1,
                "loss": avg_epoch_loss,
                "global_step": self.global_step
            })

            # Early stopping check (triggered inside batch loop)
            if no_improvement_count >= early_stopping_patience:
                break

        # Final evaluation
        final_metrics = self._evaluate(val_loader, max_samples=max_eval_samples)
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"{'='*60}")
        print(f"Final CER: {final_metrics['cer']:.4f}")
        print(f"Final WER: {final_metrics['wer']:.4f}")
        print(f"Best CER: {self.best_cer:.4f}")
        print(f"{'='*60}")

        # Save final model
        self._save_checkpoint("final")

        if writer:
            writer.close()

        return {
            "final_cer": final_metrics["cer"],
            "final_wer": final_metrics["wer"],
            "best_cer": self.best_cer,
            "total_steps": self.global_step,
            "training_history": self.training_history
        }

    def _evaluate(self, val_loader, max_samples: int = 50) -> Dict[str, float]:
        """Run evaluation on validation set."""
        self.model.eval()

        # Clear MPS cache before evaluation to prevent OOM
        if self.device.type == "mps":
            torch.mps.empty_cache()

        all_predictions = []
        all_references = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating", leave=False)):
                # Limit evaluation samples to prevent OOM
                if batch_idx * val_loader.batch_size >= max_samples:
                    break

                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"]

                # Generate predictions with reduced max_length to save memory
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=64
                )

                # Decode predictions and references
                predictions = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )

                # Decode references (replace -100 with pad token)
                labels[labels == -100] = self.processor.tokenizer.pad_token_id
                references = self.processor.batch_decode(
                    labels,
                    skip_special_tokens=True
                )

                all_predictions.extend(predictions)
                all_references.extend(references)

        # Calculate metrics
        cer = calculate_cer(all_predictions, all_references)
        wer = calculate_wer(all_predictions, all_references)

        self.model.train()

        # Clear MPS cache after evaluation
        if self.device.type == "mps":
            torch.mps.empty_cache()

        return {"cer": cer, "wer": wer}

    def _save_checkpoint(self, name: str):
        """Save a checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model and processor
        self.model.save_pretrained(checkpoint_dir)
        self.processor.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "best_cer": self.best_cer,
            "training_history": self.training_history
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"Saved checkpoint: {checkpoint_dir}")

    def export_onnx(self, output_path: str):
        """Export model to ONNX format for kwikread integration."""
        print(f"Exporting model to ONNX: {output_path}")

        try:
            from optimum.onnxruntime import ORTModelForVision2Seq

            ort_model = ORTModelForVision2Seq.from_pretrained(
                self.output_dir / "best",
                export=True
            )
            ort_model.save_pretrained(output_path)
            print(f"ONNX export complete: {output_path}")

        except Exception as e:
            print(f"ONNX export failed: {e}")
            print("You may need to install: pip install optimum[onnxruntime]")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on handwriting data")

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to training data directory")
    parser.add_argument("--corrections_file", type=str, default=None,
                        help="Path to kwikread corrections.json (optional)")

    # Model arguments
    parser.add_argument("--model_name", type=str,
                        default="microsoft/trocr-base-handwritten",
                        help="Pretrained model name")
    parser.add_argument("--output_dir", type=str,
                        default="./checkpoints",
                        help="Output directory for checkpoints")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm")

    # Evaluation arguments
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--early_stopping", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--max_eval_samples", type=int, default=20,
                        help="Maximum samples for evaluation (reduces memory usage)")
    parser.add_argument("--eval_batch_size", type=int, default=1,
                        help="Batch size for evaluation (smaller to prevent OOM)")

    # Other arguments
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device to use")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="TensorBoard log directory")
    parser.add_argument("--export_onnx", type=str, default=None,
                        help="Export to ONNX after training")

    # Data augmentation arguments
    parser.add_argument("--augment", type=bool, default=True,
                        help="Enable data augmentation")
    parser.add_argument("--augment_strength", type=str, default="medium",
                        choices=["light", "medium", "heavy"],
                        help="Augmentation strength")

    # Config file (overrides command line)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key):
                # Convert numeric strings (especially scientific notation like 1e-5)
                if isinstance(value, str):
                    try:
                        value = float(value)
                        # Convert to int if it's a whole number and the arg expects int
                        if value.is_integer() and key in ['epochs', 'batch_size', 'warmup_steps',
                                                          'gradient_accumulation', 'eval_steps',
                                                          'save_steps', 'early_stopping',
                                                          'max_eval_samples', 'eval_batch_size',
                                                          'num_workers']:
                            value = int(value)
                    except ValueError:
                        pass  # Keep as string if not a number
                setattr(args, key, value)

    # Initialize trainer
    trainer = TrOCRTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        device=args.device
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        processor=trainer.processor,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        augment_train=args.augment,
        augment_strength=args.augment_strength
    )

    # Train
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_grad_norm=args.max_grad_norm,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        early_stopping_patience=args.early_stopping,
        use_amp=not args.no_amp,
        log_dir=args.log_dir,
        max_eval_samples=args.max_eval_samples
    )

    # Export to ONNX if requested
    if args.export_onnx:
        trainer.export_onnx(args.export_onnx)

    # Save results
    results_path = Path(args.output_dir) / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
