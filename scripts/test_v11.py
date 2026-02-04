#!/usr/bin/env python3
"""Test Hailo DFC translation for v11 model (RoPE fix + Input Promotion)."""
from hailo_sdk_client import ClientRunner
import sys
import sys
import os
import traceback

# Path to the v11 model
MODEL_PATH = os.path.expanduser("~/lfm_compile/lfm2.5_fp16_fixed_v11_hailo.onnx")

# Check if file exists relative to repo root if not in home
if not os.path.exists(MODEL_PATH) and os.path.exists(f"models/{os.path.basename(MODEL_PATH)}"):
     MODEL_PATH = os.path.abspath(f"models/{os.path.basename(MODEL_PATH)}")

print(f"Loading v11 model from: {MODEL_PATH}")
print("Initializing Hailo ClientRunner for hailo10h...")
runner = ClientRunner(hw_arch="hailo10h")

print("Starting ONNX translation to Hailo IR...")
try:
    hn = runner.translate_onnx_model(
        MODEL_PATH,
        net_name="lfm2_5_v11"
    )
    print("\n" + "="*50)
    print("✅ TRANSLATION SUCCESS!")
    print("="*50)
    print(f"Result type: {type(hn)}")
    
    # Save the HN file
    hn_path = os.path.expanduser("~/lfm_compile/lfm2_5_v11.hn")
    os.makedirs(os.path.dirname(hn_path), exist_ok=True)
    runner.save_hn(hn_path)
    print(f"Saved Hailo Network to: {hn_path}")
    
except Exception as e:
    print("\n" + "="*50)
    print("❌ TRANSLATION FAILED")
    print("="*50)
    print(f"Error: {e}")
    traceback.print_exc()
    sys.exit(1)
