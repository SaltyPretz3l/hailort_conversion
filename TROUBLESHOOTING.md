# Troubleshooting - LFM2.5 to Hailo Conversion

> **Last Updated**: 2026-02-04
> **Context**: Debugging Hailo Dataflow Compiler (DFC) v3.28.0 conversion of Llama-based model.

## 1. `IndexError: list index out of range` in `_create_layer`

**Symptoms**:
- Compilation fails during `translate_onnx_model`.
- Stack trace points to `hailo_sdk_client/model_translator/edge_nn_translator.py`, typically in `axes_to_nhwc` or similar dimension mapping functions.

**Causes**:
1.  **Rank Mismatch**: The Hailo parser expects 4D inputs `[B, H, W, C]` (or comparable) for many operations. If an ONNX node produces a 3D tensor `[B, S, Hidden]`, the internal mapper may fail when trying to access index 3.
2.  **Unsqueeze vs Reshape**: Using `Unsqueeze` to insert dimensions into a Rank 3 tensor to make it Rank 4 can sometimes confuse the parser if it cannot statically infer the new shape immediately.
3.  **Ambiguous Dimensions**: Using `0` in `Reshape` ops at non-standard positions can cause `ShapeInferenceError: Invalid position of 0`.

**Solutions**:
- **Explicit Reshape**: Instead of `Unsqueeze`, use `Reshape` with a fully specified shape or correct usage of `-1`.
- **Rank 4 Promotion**: Ensure inputs to complex layers (like RoPE or Attention) are reshaped to 4D `[B, S, Heads, HeadDim]` explicitly before processing.
- **Valid Reshape Indices**: Use `-1` to infer dimensions. Avoid `0` unless it refers to a dimension being copied (which is standard ONNX behavior but can be flaky in some parsers if not strictly adhered to).

## 2. `ShapeInferenceError` in `Reshape`

**Symptoms**:
- Error message: `Op (Reshape) [ShapeInferenceError] Invalid position of 0`.

**Cause**:
- In ONNX `Reshape`, a `0` in the shape tensor means "copy the dimension from input". This is only valid if the input dimension exists at that index. If reshaping `[B, S, Hidden]` to `[B, S, 1, Hidden]`, using `0` at index 2 is invalid because the input only has 3 dims.

**Solution**:
- Use `-1` for the inferred dimension.
- Example: Target `[B, S, 1, HeadDim]` from `[B, S, HeadDim]` -> Use shape `[0, 0, 1, -1]` (Copy 0, Copy 1, Insert 1, Infer 3).

## 3. Parser Crash on `model.embed_tokens.weight`

**Symptoms**:
- Hailo parser crashes or hangs when `model.embed_tokens.weight` is a graph initializer (constant).

**Cause**:
- Large weights as initializers can overwhelm the parser's constant folding or graph optimization steps.

**Solution**:
- **Promote to Input**: Convert the large weight tensor from an initializer to a graph input. This forces the compiler to treat it as data flow rather than a baked-in constant, bypassing the bottleneck.

## 4. `is_null_transpose_near_torch_tile` Error

**Symptoms**:
- Compilation fails with an assertion or error related to `is_null_transpose_near_torch_tile`.

**Cause**:
- Specific pattern of `Unsqueeze` -> `Tile` -> `Reshape` used in GQA (Group Query Attention) decomposition.
- **Also observed in RoPE**: The `Reshape` -> `Transpose` pattern for RoPE heads can trigger this if graph optimization passes get confused by dynamic shapes or missing inputs.

**Traceback**:
```text
File ".../onnx_translator/onnx_graph.py", line 5369, in is_null_transpose_near_torch_tile
    input_shape = self.get_input_shapes(convert_to_nhwc=False)[0]
IndexError: list index out of range
```

**Solution**:
- Rewrite decomposition to use `Split` -> `Concat` instead of `Tile`. Split the heads, repeat them via list duplication, and concat them back.
