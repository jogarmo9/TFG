#!/usr/bin/env python3
"""Download Silero VAD model"""
import urllib.request
from pathlib import Path

model_path = Path("d:/Uni/TFG/TFG_claude/TFG/models/silero_vad.onnx")
# Try different URLs for Silero VAD ONNX model
urls = [
    "https://github.com/snakers4/silero-vad/releases/download/v3.1/silero_vad.onnx",
    "https://huggingface.co/snakers4/silero-vad/resolve/main/silero_vad_en.onnx",
]
model_url = urls[0]  # Try first URL

if model_path.exists():
    print(f"✓ {model_path.name} already exists")
else:
    print(f"Downloading Silero VAD model...")
    try:
        urllib.request.urlretrieve(model_url, str(model_path))
        print(f"✓ Downloaded to {model_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
