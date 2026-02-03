"""
Hailo Dataflow Compiler (DFC) script for LFM2.5-Thinking.

Implements:
- Mixed-precision quantization (INT4 for LIV, INT8 for GQA)
- Memory allocation optimization (SRAM for LIV, DDR spillover for GQA)
- Context window configuration

IMPORTANT: This script must be run in a WSL2/Linux environment
with the Hailo SDK installed from the Hailo Developer Zone.

Usage:
    python quantize_and_compile.py --model ./models/lfm2.5_thinking_decomposed.onnx
"""
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import json
import sys

# Check for Hailo SDK availability
HAILO_AVAILABLE = False
try:
    from hailo_sdk_client import ClientRunner
    HAILO_AVAILABLE = True
except ImportError:
    print("=" * 60)
    print("WARNING: hailo_sdk_client not available")
    print("=" * 60)
    print("This script requires the Hailo SDK which only runs on Linux/WSL2.")
    print("Install from: https://hailo.ai/developer-zone/")
    print()
    print("For Windows users:")
    print("  1. Install WSL2 with Ubuntu 20.04/22.04")
    print("  2. Install Hailo SDK in WSL2")
    print("  3. Run this script from WSL2")
    print()


# Configuration
DEFAULT_CONFIG = {
    "hw_arch": "hailo10",
    "context_limit": 4096,
    "quantization": {
        # LIV layers (0-9): More aggressive quantization
        # These are feature extractors, robust to noise
        "liv_layers": {
            "pattern": ["liv_*", "conv1d_*", "layer_0_*", "layer_1_*", "layer_2_*",
                       "layer_3_*", "layer_4_*", "layer_5_*", "layer_6_*", 
                       "layer_7_*", "layer_8_*", "layer_9_*"],
            "weights_bits": 4,
            "activations_bits": 8
        },
        # GQA/Attention layers (10-15): Preserve precision for reasoning
        "gqa_layers": {
            "pattern": ["gqa_*", "attention_*", "layer_10_*", "layer_11_*",
                       "layer_12_*", "layer_13_*", "layer_14_*", "layer_15_*"],
            "weights_bits": 8,
            "activations_bits": 8
        }
    },
    "memory_allocation": {
        # LIV layers benefit from fast SRAM access
        "sram_layers": ["liv_*", "conv1d_*"],
        # GQA KV-cache can spill to DDR if needed
        "ddr_spillover_layers": ["gqa_*", "attention_*", "kv_cache_*"]
    }
}


def create_quantization_config(config: Dict[str, Any]) -> str:
    """
    Create Hailo quantization configuration script.
    
    This generates the HAR (Hailo Archive) quantization commands.
    """
    quant_script = []
    
    # LIV layer quantization
    liv_config = config["quantization"]["liv_layers"]
    for pattern in liv_config["pattern"]:
        quant_script.append(
            f"quantization_param({pattern}, precision_mode=mixed, "
            f"weights_precision={liv_config['weights_bits']}, "
            f"activations_precision={liv_config['activations_bits']})"
        )
    
    # GQA layer quantization
    gqa_config = config["quantization"]["gqa_layers"]
    for pattern in gqa_config["pattern"]:
        quant_script.append(
            f"quantization_param({pattern}, precision_mode=mixed, "
            f"weights_precision={gqa_config['weights_bits']}, "
            f"activations_precision={gqa_config['activations_bits']})"
        )
    
    return "\n".join(quant_script)


def create_allocator_config(config: Dict[str, Any]) -> str:
    """
    Create Hailo memory allocation configuration.
    
    Optimizes SRAM vs DDR allocation for the LFM architecture:
    - LIV layers: Force to SRAM for fast conv operations
    - GQA layers: Allow DDR spillover for KV-cache
    """
    alloc_script = []
    
    # Force LIV layers to SRAM
    for pattern in config["memory_allocation"]["sram_layers"]:
        alloc_script.append(f"allocator_param(layers=['{pattern}'], loc='sram')")
    
    # Allow GQA KV-cache to use DDR if SRAM is exhausted
    for pattern in config["memory_allocation"]["ddr_spillover_layers"]:
        alloc_script.append(f"allocator_param(layers=['{pattern}'], loc='ddr_if_needed')")
    
    # Set context limit
    alloc_script.append(f"context_param(max_tokens={config['context_limit']})")
    
    return "\n".join(alloc_script)


def validate_model_for_hailo(model_path: str) -> bool:
    """
    Validate ONNX model before Hailo compilation.
    """
    try:
        import onnx
        
        print(f"Validating model: {model_path}")
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        
        # Check for unsupported operators
        unsupported = []
        for node in model.graph.node:
            if "RecursiveScript" in node.op_type or "LIV" in node.op_type:
                unsupported.append(f"{node.name}: {node.op_type}")
        
        if unsupported:
            print("⚠️ Found potentially unsupported operators:")
            for op in unsupported[:5]:
                print(f"    {op}")
            print("\nRun liv_decomposition.py first to decompose these operators.")
            return False
        
        print("✓ Model validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Model validation failed: {e}")
        return False


def run_mock_compilation(
    model_path: str,
    calib_path: str,
    output_path: str,
    config: Dict[str, Any]
):
    """
    Mock compilation for development/testing without Hailo SDK.
    """
    print("\n" + "=" * 60)
    print("MOCK COMPILATION MODE")
    print("=" * 60)
    print("The following operations would be performed with Hailo SDK:")
    print()
    
    print("1. Load Model:")
    print(f"   runner.load_model_script('{model_path}')")
    print()
    
    print("2. Quantization Configuration:")
    quant_config = create_quantization_config(config)
    for line in quant_config.split("\n")[:5]:
        print(f"   {line}")
    print("   ...")
    print()
    
    print("3. Load Calibration Data:")
    print(f"   calib_data = np.load('{calib_path}')")
    print()
    
    print("4. Optimize with Mixed Precision:")
    print("   runner.optimize(calib_data, precision_mode='manual', config=quant_config)")
    print()
    
    print("5. Memory Allocation:")
    alloc_config = create_allocator_config(config)
    for line in alloc_config.split("\n"):
        print(f"   {line}")
    print()
    
    print("6. Compile to HEF:")
    print(f"   hef = runner.compile(allocator_script=allocator_script)")
    print(f"   runner.save_hef('{output_path}')")
    print()
    
    # Create placeholder files for pipeline testing
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save compilation config for reference
    config_path = Path(output_path).with_suffix(".json")
    with open(config_path, "w") as f:
        json.dump({
            "model_path": model_path,
            "calib_path": calib_path,
            "output_path": output_path,
            "config": config,
            "status": "mock_compilation",
            "note": "Run in WSL2 with Hailo SDK for actual compilation"
        }, f, indent=2)
    
    print(f"✓ Mock compilation complete")
    print(f"  Config saved to: {config_path}")
    print(f"\nTo complete actual compilation:")
    print(f"  1. Transfer files to WSL2/Linux environment")
    print(f"  2. Install Hailo SDK")
    print(f"  3. Run this script again")


def run_hailo_compilation(
    model_path: str,
    calib_path: str,
    output_path: str,
    config: Dict[str, Any]
):
    """
    Run actual Hailo compilation with SDK.
    """
    import numpy as np
    
    print("\n" + "=" * 60)
    print("HAILO DATAFLOW COMPILATION")
    print("=" * 60)
    
    # Initialize Hailo client
    print(f"\nInitializing ClientRunner for {config['hw_arch']}...")
    runner = ClientRunner(hw_arch=config['hw_arch'])
    
    # Load model
    print(f"Loading model: {model_path}")
    runner.load_model_script(model_path)
    
    # Load calibration data
    print(f"Loading calibration data: {calib_path}")
    calib_data = np.load(calib_path)
    print(f"  Calibration shape: {calib_data.shape}")
    
    # Create quantization config
    print("\nApplying mixed-precision quantization...")
    quant_config = {
        'layer_0_to_9': {
            'weights': config['quantization']['liv_layers']['weights_bits'],
            'activations': config['quantization']['liv_layers']['activations_bits']
        },
        'layer_10_to_15': {
            'weights': config['quantization']['gqa_layers']['weights_bits'],
            'activations': config['quantization']['gqa_layers']['activations_bits']
        }
    }
    
    # Run optimization
    runner.optimize(calib_data, precision_mode='manual', config=quant_config)
    
    # Create allocator script
    allocator_script = create_allocator_config(config)
    
    # Compile
    print("\nCompiling for Hailo-10H...")
    hef = runner.compile(allocator_script=allocator_script)
    
    # Save HEF
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    runner.save_hef(output_path)
    
    print(f"\n✓ Compilation complete!")
    print(f"  HEF saved to: {output_path}")
    
    # Save metadata
    meta_path = Path(output_path).with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "model": model_path,
            "hw_arch": config['hw_arch'],
            "context_limit": config['context_limit'],
            "quantization": config['quantization'],
            "status": "compiled"
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Compile LFM2.5-Thinking for Hailo-10H NPU"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/lfm2.5_thinking_decomposed.onnx",
        help="Path to decomposed ONNX model"
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="./calibration/calib_thinking.npy",
        help="Path to calibration data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output/lfm2.5_thinking_h10.hef",
        help="Output path for HEF file"
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        default=4096,
        help="Context window size (default: 4096)"
    )
    parser.add_argument(
        "--hw-arch",
        type=str,
        default="hailo10",
        choices=["hailo8", "hailo10"],
        help="Target hardware architecture"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip model validation step"
    )
    parser.add_argument(
        "--force-mock",
        action="store_true",
        help="Force mock compilation even if SDK is available"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LFM2.5-Thinking Hailo Compilation")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Calibration: {args.calibration}")
    print(f"  Output: {args.output}")
    print(f"  Context Limit: {args.context_limit}")
    print(f"  HW Architecture: {args.hw_arch}")
    print(f"  Hailo SDK Available: {HAILO_AVAILABLE}")
    print()
    
    # Update config with CLI args
    config = DEFAULT_CONFIG.copy()
    config["hw_arch"] = args.hw_arch
    config["context_limit"] = args.context_limit
    
    # Validate model
    if not args.skip_validation:
        if not validate_model_for_hailo(args.model):
            print("\nModel validation failed. Please fix issues before compiling.")
            sys.exit(1)
    
    # Check calibration data
    if not Path(args.calibration).exists():
        print(f"\n⚠️ Calibration data not found: {args.calibration}")
        print("Run generate_calibration_data.py first.")
        print("\nContinuing with mock compilation...")
        args.force_mock = True
    
    # Run compilation
    if HAILO_AVAILABLE and not args.force_mock:
        run_hailo_compilation(
            args.model,
            args.calibration,
            args.output,
            config
        )
    else:
        run_mock_compilation(
            args.model,
            args.calibration,
            args.output,
            config
        )


if __name__ == "__main__":
    main()
