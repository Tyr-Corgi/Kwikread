#!/usr/bin/env python3
"""
Export TrOCR Decoder to ONNX Format

Exports the decoder model for use with kwikread. This creates a simple
decoder without KV cache (slower but compatible).
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel, TrOCRProcessor


class DecoderWrapper(nn.Module):
    """Wrapper for TrOCR decoder that handles encoder_hidden_states input."""

    def __init__(self, decoder, encoder_hidden_size=768):
        super().__init__()
        self.decoder = decoder
        self.encoder_hidden_size = encoder_hidden_size

    def forward(self, input_ids, encoder_hidden_states):
        # Create attention mask for encoder outputs
        batch_size = encoder_hidden_states.shape[0]
        seq_len = encoder_hidden_states.shape[1]
        encoder_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=encoder_hidden_states.device)

        # Run decoder
        outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True
        )

        return outputs.logits


def export_decoder(checkpoint_path: str, output_dir: str, opset_version: int = 14):
    """Export TrOCR decoder to ONNX format."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {checkpoint_path}")
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)
    model.eval()

    # Get model config
    encoder_hidden_size = model.config.encoder.hidden_size
    print(f"Encoder hidden size: {encoder_hidden_size}")

    # Create decoder wrapper
    decoder_wrapper = DecoderWrapper(model.decoder, encoder_hidden_size)
    decoder_wrapper.eval()

    # Create dummy inputs matching encoder output shape
    # TrOCR encoder outputs (batch, 577, hidden_size) for 384x384 input
    # 577 = (384/16)^2 + 1 = 576 patches + 1 CLS token
    batch_size = 1
    encoder_seq_len = 577

    dummy_input_ids = torch.tensor([[model.config.decoder_start_token_id]], dtype=torch.long)
    dummy_encoder_hidden = torch.randn(batch_size, encoder_seq_len, encoder_hidden_size)

    decoder_path = output_path / "decoder_model.onnx"
    print(f"Exporting decoder to: {decoder_path}")

    # Export decoder
    torch.onnx.export(
        decoder_wrapper,
        (dummy_input_ids, dummy_encoder_hidden),
        str(decoder_path),
        input_names=["input_ids", "encoder_hidden_states"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "encoder_hidden_states": {0: "batch_size", 1: "encoder_sequence"},
            "logits": {0: "batch_size", 1: "sequence"}
        },
        opset_version=opset_version,
        dynamo=False
    )

    # Verify export
    import onnx
    onnx_model = onnx.load(str(decoder_path))
    onnx.checker.check_model(onnx_model)
    print(f"Decoder exported successfully: {decoder_path}")
    print(f"File size: {decoder_path.stat().st_size / 1024 / 1024:.1f} MB")

    return decoder_path


def main():
    parser = argparse.ArgumentParser(description="Export TrOCR decoder to ONNX")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to checkpoint")
    parser.add_argument("--output", "-o", default="./models", help="Output directory")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")

    args = parser.parse_args()
    export_decoder(args.checkpoint, args.output, args.opset)


if __name__ == "__main__":
    main()
