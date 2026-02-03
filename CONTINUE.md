# Continuation Guide - LFM2.5 to Hailo Conversion

> **Last Updated**: 2026-02-03  
> **Previous Environment**: Raspberry Pi (ARM64 - incompatible with Hailo DFC)  
> **Next Environment**: x86_64 Linux/WSL2 with Hailo SDK

## Current Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Export | ✅ Complete | Model verified compatible - no surgery needed |
| Phase 2: Calibration | ✅ Complete | 512 GSM8k samples generated |
| Phase 3: Compilation | ⚠️ **BLOCKED** | Requires `hailo_sdk_client` on x86_64 |
| Phase 4: Verification | ❌ Pending | Awaiting HEF file |

## Important Findings

1. **Graph Compatibility**: The LFM2.5 ONNX export from HuggingFace is **already compatible** with Hailo DFC. The `liv_decomposition.py --dry-run` confirmed no graph surgery is needed.

2. **ONNX Compatibility Patch**: The `scripts/liv_decomposition.py` file has been patched to work with newer `onnx` versions (the `float32_to_bfloat16` function was missing in recent releases).

3. **Model Size**: The full model is ~4.6GB with external data files. Download takes ~10-15 minutes depending on network speed.

## Next Steps for WSL2/x86_64

### 1. Clone and Setup Environment

```bash
git clone https://github.com/SaltyPretz3l/LFM-Convert-to-HailoRT.git
cd LFM-Convert-to-HailoRT

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Hailo SDK

Download from [Hailo Developer Zone](https://hailo.ai/developer-zone/) and install:

```bash
pip install hailo_sdk_client-<version>-py3-none-linux_x86_64.whl
pip install hailo_platform-<version>-py3-none-linux_x86_64.whl
```

### 3. Download Model & Generate Calibration Data

```bash
# Download ONNX model (~4.6GB, takes 10-15 min)
python scripts/export_lfm_onnx.py --output-dir ./models

# Generate calibration data (512 GSM8k samples)
python scripts/generate_calibration_data.py --samples 512 --output ./calibration/calib_thinking.npy
```

### 4. Compile for Hailo-10H

```bash
python scripts/quantize_and_compile.py \
    --model ./models/repo/onnx/model.onnx \
    --calibration ./calibration/calib_thinking.npy \
    --output ./output/lfm2.5_thinking_h10.hef \
    --skip-validation
```

> **Note**: Use `--skip-validation` because loading the full 4.6GB model for validation can cause OOM on systems with limited RAM.

### 5. Test Runtime (if Hailo device available)

```bash
python scripts/hailo_runtime.py --hef ./output/lfm2.5_thinking_h10.hef
```

## Quantization Configuration

The compilation uses a mixed-precision strategy optimized for the LFM2.5 architecture:

- **LIV Layers (0-9)**: INT4 weights, INT8 activations (SRAM allocation)
- **GQA Layers (10-15)**: INT8 weights, INT8 activations (DDR spillover allowed)

## Troubleshooting

### OOM during validation
Use `--skip-validation` flag. The model was already verified compatible on the previous machine.

### `onnx_graphsurgeon` import error
The patch in `liv_decomposition.py` should handle this. If issues persist:
```bash
pip install "onnx<1.16.0" onnx-graphsurgeon
```

### Download hangs or fails
Set a HuggingFace token for faster downloads:
```bash
export HF_TOKEN=your_token_here
```

## Files to Generate

After completing the steps above, you should have:

| File | Description |
|------|-------------|
| `models/repo/onnx/model.onnx` | Base ONNX model |
| `models/repo/onnx/model.onnx_data*` | External weight files |
| `calibration/calib_thinking.npy` | Calibration dataset |
| `output/lfm2.5_thinking_h10.hef` | **Final compiled HEF file** |

## Questions?

Refer to the main [README.md](./README.md) for architecture details and the [Protocol Document](./Protocol_LFM2.5_ThinkingConversion.md) for the full conversion protocol.
