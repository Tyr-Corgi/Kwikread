#!/usr/bin/env python3
"""
Export Fine-tuned TrOCR Model to ONNX Format

Exports the trained model to ONNX format for integration with kwikread's
C# ONNX Runtime inference engine.

Usage:
    python export_onnx.py --checkpoint ./checkpoints/best --output ./models
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor


def export_trocr_to_onnx(
    checkpoint_path: str,
    output_dir: str,
    quantize: bool = True,
    opset_version: int = 14
):
    """
    Export TrOCR model to ONNX format.

    Args:
        checkpoint_path: Path to the trained checkpoint
        output_dir: Directory to save ONNX files
        quantize: Whether to quantize the model (INT8)
        opset_version: ONNX opset version
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {checkpoint_path}")

    # Load model and processor
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)
    processor = TrOCRProcessor.from_pretrained(checkpoint_path)

    model.eval()

    print("Exporting to ONNX format...")

    try:
        # Method 1: Use Optimum library (recommended)
        from optimum.onnxruntime import ORTModelForVision2Seq

        print("Using Optimum for export...")

        # Export with optimum
        ort_model = ORTModelForVision2Seq.from_pretrained(
            checkpoint_path,
            export=True
        )

        # Save ONNX model
        onnx_path = output_path / "onnx"
        ort_model.save_pretrained(onnx_path)
        processor.save_pretrained(onnx_path)

        print(f"ONNX model saved to: {onnx_path}")

        # Quantize if requested
        if quantize:
            print("Quantizing model (INT8)...")
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig

            quantizer = ORTQuantizer.from_pretrained(onnx_path)
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)

            quantized_path = output_path / "onnx_quantized"
            quantizer.quantize(save_dir=quantized_path, quantization_config=qconfig)
            print(f"Quantized model saved to: {quantized_path}")

    except ImportError:
        # Method 2: Manual ONNX export
        print("Optimum not available, using manual export...")
        _manual_onnx_export(model, processor, output_path, opset_version, quantize)

    # Copy vocab.json for kwikread (token -> id format)
    vocab_path = output_path / "vocab.json"
    if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'vocab'):
        import json
        vocab = processor.tokenizer.get_vocab()
        # kwikread expects {token: id} format
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False)
        print(f"Vocabulary saved to: {vocab_path}")

    print("\nExport complete!")
    print("\nTo use with kwikread, copy these files to Kwikread/Models/:")
    print(f"  - encoder_model.onnx")
    print(f"  - decoder_model_merged.onnx")
    print(f"  - vocab.json")


def _manual_onnx_export(model, processor, output_path, opset_version, quantize):
    """
    Manual ONNX export without Optimum library.
    """
    import onnx

    # Create dummy inputs
    dummy_pixel_values = torch.randn(1, 3, 384, 384)
    dummy_decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

    # Export encoder
    encoder_path = output_path / "encoder_model.onnx"
    print(f"Exporting encoder to: {encoder_path}")

    # Use dynamo=False to avoid onnxscript issues with Python 3.14
    torch.onnx.export(
        model.encoder,
        dummy_pixel_values,
        str(encoder_path),
        input_names=["pixel_values"],
        output_names=["encoder_hidden_states"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "encoder_hidden_states": {0: "batch_size"}
        },
        opset_version=opset_version,
        dynamo=False
    )

    # Export decoder (more complex due to KV cache)
    # For simplicity, export without KV cache - less efficient but simpler
    print("Note: Decoder export with KV cache requires Optimum library")
    print("Installing optimum is recommended: pip install optimum[onnxruntime]")

    # Quantize if requested and onnxruntime available
    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            print("Quantizing encoder...")
            quantized_encoder = output_path / "encoder_model_quantized.onnx"
            quantize_dynamic(
                str(encoder_path),
                str(quantized_encoder),
                weight_type=QuantType.QUInt8
            )
            print(f"Quantized encoder saved to: {quantized_encoder}")
        except ImportError:
            print("Skipping quantization - onnxruntime not available")
            print("Install onnxruntime to enable: pip install onnxruntime")


def integrate_with_kwikread(onnx_dir: str, kwikread_models_dir: str):
    """
    Copy exported ONNX files to kwikread Models directory.

    Args:
        onnx_dir: Directory containing exported ONNX files
        kwikread_models_dir: Path to Kwikread/Models directory
    """
    onnx_path = Path(onnx_dir)
    models_path = Path(kwikread_models_dir)

    if not models_path.exists():
        print(f"Creating models directory: {models_path}")
        models_path.mkdir(parents=True, exist_ok=True)

    # File mappings
    files_to_copy = [
        ("encoder_model.onnx", "encoder_model_quantized.onnx"),
        ("decoder_model_merged.onnx", "decoder_model_merged_quantized.onnx"),
        ("vocab.json", "vocab.json")
    ]

    # Also check quantized versions
    quantized_files = [
        ("encoder_model_quantized.onnx", "encoder_model_quantized.onnx"),
        ("decoder_model_merged_quantized.onnx", "decoder_model_merged_quantized.onnx"),
    ]

    print(f"\nCopying ONNX files to: {models_path}")

    for src_name, dst_name in files_to_copy + quantized_files:
        src = onnx_path / src_name
        if src.exists():
            dst = models_path / dst_name
            shutil.copy2(src, dst)
            print(f"  Copied: {src_name} -> {dst_name}")

    print("\nIntegration complete!")
    print("Rebuild kwikread to use the fine-tuned model:")
    print("  cd Kwikread && dotnet build")


def main():
    parser = argparse.ArgumentParser(
        description="Export fine-tuned TrOCR to ONNX for kwikread"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        required=True,
        help="Path to trained checkpoint directory"
    )
    parser.add_argument(
        "--output", "-o",
        default="./models",
        help="Output directory for ONNX files"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip INT8 quantization"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--integrate",
        type=str,
        default=None,
        help="Path to Kwikread/Models directory to copy files"
    )

    args = parser.parse_args()

    # Export to ONNX
    export_trocr_to_onnx(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        quantize=not args.no_quantize,
        opset_version=args.opset
    )

    # Integrate with kwikread if requested
    if args.integrate:
        integrate_with_kwikread(args.output, args.integrate)


if __name__ == "__main__":
    main()
