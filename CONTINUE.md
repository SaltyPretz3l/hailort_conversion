# Continuation Guide - LFM2.5 to Hailo Conversion

> **Last Updated**: 2026-02-04 15:15
> **Status**: 🔴 **BLOCKED** - v11 RoPE optimization failed.
> **Current Step**: Pivot to alternative strategies (v7 verification or new decomposition).
> **Docs**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for error details.

## 📉 Current Status
We attempted to fix the RoPE decomposition in **v11** using both dynamic and static shape logic. All attempts failed with `IndexError: list index out of range` in `hailo_sdk_client`. The error is located in `is_null_transpose_near_torch_tile`, suggesting the parser is crashing on the RoPE subgraph structure.

### Recent Fixes & Attempts
| Model Version | Change | Result |
|---|---|---|
| **v11 (A)** | Initial v11 | **Failed**: `IndexError`. |
| **v11 (B)** | Dynamic Shape Logic | **Failed**: `IndexError`. |
| **v11 (C)** | Static Name-based Heads | **Failed**: `IndexError`. |

## 🔍 The Issue
The `IndexError` persists regardless of how we compute the shapes. It appears to be a bug in the Hailo DFC parser when handling the specific `Reshape`->`Transpose` pattern used in our RoPE decomposition, posssibly related to how it traces inputs for Torch tiling checks.

## 📋 Next Steps
1.  **Revert to v7**: The v11 path is currently dead. We should focus on verifying if **v7** (which compiled successfully) is functionally correct despite its unoptimized RoPE.
2.  **Isolate RoPE**: Create a minimal reproduction script with ONLY a RoPE layer to isolate the parser bug.
3.  **Contact Support**: If isolated, report to Hailo.

## Files
*   `models/lfm2.5_fp16_fixed_v11_hailo.onnx`: Failed model.
*   `scripts/test_v11.py`: Test script (now prints traceback).
*   `~/lfm_compile/compilation.log` (WSL): Full error details.
