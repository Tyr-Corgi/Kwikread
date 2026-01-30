#!/usr/bin/env python3
"""Download trocr-base-handwritten ONNX model from Hugging Face"""

import os
import urllib.request
import json

# Xenova trocr-base-handwritten
MODEL_ID = "Xenova/trocr-base-handwritten"
BASE_URL = f"https://huggingface.co/{MODEL_ID}/resolve/main/onnx"

FILES = [
    "encoder_model.onnx",
    "decoder_model.onnx", 
    "decoder_model_merged.onnx",  # Try this one
]

CONFIG_FILES = [
    "config.json",
    "vocab.json",
    "tokenizer.json",
    "generation_config.json",
]

OUTPUT_DIR = "models/onnx_base"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_file(url, dest):
    print(f"Downloading {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  -> Saved to {dest} ({os.path.getsize(dest) / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"  -> Failed: {e}")
        return False

# First download config to check model architecture
config_url = f"https://huggingface.co/{MODEL_ID}/resolve/main/config.json"
config_path = os.path.join(OUTPUT_DIR, "config.json")
download_file(config_url, config_path)

# Read and print config
with open(config_path) as f:
    config = json.load(f)
    print("\nModel config:")
    print(f"  Encoder hidden_size: {config.get('encoder', {}).get('hidden_size')}")
    print(f"  Decoder d_model: {config.get('decoder', {}).get('d_model')}")
    print(f"  Decoder layers: {config.get('decoder', {}).get('decoder_layers')}")
    print(f"  decoder_start_token_id: {config.get('decoder_start_token_id')}")
    print(f"  Vocab size: {config.get('vocab_size')}")
