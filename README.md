# LFM2.5-1.2B-Thinking → Hailo-10H NPU Conversion

Convert LiquidAI's LFM2.5-1.2B-Thinking model to run on Hailo-10H Neural Processing Unit.

## Overview

This project provides a complete pipeline to convert the [LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking) hybrid language model for edge deployment on Hailo-10H NPU with 40 TOPS INT4 compute.

### Architecture

LFM2.5 uses a hybrid architecture:
- **Layers 0-9**: Linear Input Varying (LIV) blocks with Conv1D + Gating
- **Layers 10-15**: Grouped-Query Attention (GQA) for reasoning

### Features

- ✅ ONNX model export with Conv1D view for LIV blocks
- ✅ Graph surgery for operator decomposition
- ✅ Mixed-precision quantization (INT4/INT8)
- ✅ Memory-optimized compilation (SRAM/DDR allocation)
- ✅ Runtime wrapper with `<thinking>` mode support

## Requirements

### System Requirements

- **Linux**: Ubuntu 20.04/22.04, 64-bit
- **Python**: 3.8 - 3.12
- **Hardware**: Hailo-10H device (for final deployment)

### Software Requirements

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Hailo SDK (download from Hailo Developer Zone)
# https://hailo.ai/developer-zone/
# - hailo_sdk_client (Dataflow Compiler)
# - hailo_platform (HailoRT)
```

## Project Structure

```
LFM Conversion to HailoRT/
├── scripts/
│   ├── export_lfm_onnx.py        # ONNX model export
│   ├── liv_decomposition.py      # Graph surgery
│   ├── generate_calibration_data.py  # Calibration data
│   ├── quantize_and_compile.py   # Hailo compilation
│   └── hailo_runtime.py          # Runtime wrapper
├── models/                       # ONNX models (generated)
├── calibration/                  # Calibration data (generated)
├── output/                       # Compiled HEF files (generated)
├── requirements.txt
├── Protocol_LFM2.5_ThinkingConversion.md
└── README.md
```

## Quick Start

### Phase 1: Export & Graph Surgery

```bash
# Download ONNX model from HuggingFace
python scripts/export_lfm_onnx.py --output-dir ./models

# Check if decomposition is needed (LiquidAI export usually has this pre-done)
python scripts/liv_decomposition.py ./models/repo/onnx/model.onnx --dry-run

# If decomposition is needed (check output of above command):
# python scripts/liv_decomposition.py ./models/repo/onnx/model.onnx ./models/lfm2.5_thinking_decomposed.onnx
```

### Phase 2: Calibration & Quantization

```bash
# Generate calibration dataset (GSM8k reasoning traces)
python scripts/generate_calibration_data.py --samples 512 --output ./calibration/calib_thinking.npy
```

### Phase 3: Compile for Hailo

```bash
# Compile with mixed-precision quantization
python scripts/quantize_and_compile.py \
    --model ./models/repo/onnx/model.onnx \
    --calibration ./calibration/calib_thinking.npy \
    --output ./output/lfm2.5_thinking_h10.hef \
    --context-limit 4096 \
    --skip-validation
```

### Phase 4: Runtime Inference

```python
from scripts.hailo_runtime import LFMHailoInference

# Initialize engine
engine = LFMHailoInference("./output/lfm2.5_thinking_h10.hef")

# Generate with thinking mode support
for result in engine.generate(input_ids):
    if result.is_thinking:
        print(f"\r{result.status}", end="")
    else:
        print(result.text, end="")
```

## Quantization Strategy

| Layer Range | Type | Weights | Activations | Rationale |
|------------|------|---------|-------------|-----------|
| 0-9 | LIV Conv | INT4 | INT8 | Feature extractors, robust to noise |
| 10-15 | GQA Attention | INT8 | INT8 | Preserve reasoning pathways |

## Memory Allocation

- **LIV layers**: Forced to SRAM for fast convolution
- **GQA KV-cache**: DDR spillover allowed for context storage

## Troubleshooting

### Error: `Graph contains unsupported Op: RecursiveScript`

The LIV block was not properly flattened. Re-run the decomposition:

```bash
python scripts/liv_decomposition.py ./models/model.onnx ./models/model_decomposed.onnx
```

### Error: `Context Mismatch`

Ensure the input shape matches the ONNX export shape:

```python
# Check expected input shape
import onnx
model = onnx.load("./models/model.onnx")
print(model.graph.input[0])
```

### Error: `hailo_sdk_client not found`

Install the Hailo SDK from the [Hailo Developer Zone](https://hailo.ai/developer-zone/):

```bash
pip install hailo_sdk_client-<version>-py3-none-linux_x86_64.whl
```

## References

- [LiquidAI LFM2.5 Models](https://huggingface.co/LiquidAI)
- [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- [Protocol Document](./Protocol_LFM2.5_ThinkingConversion.md)

## License

This conversion pipeline is provided for educational purposes. Please refer to:
- LiquidAI's model license for LFM2.5
- Hailo's SDK license for the compiler and runtime
