"""
Export LFM2.5-1.2B-Thinking to ONNX format.

This script downloads the pre-exported ONNX model from HuggingFace.
The LiquidAI ONNX exports are optimized with Conv1D view for LIV blocks.

Usage:
    python export_lfm_onnx.py [--output-dir ./models]
"""
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

# Model configuration
MODEL_REPO = "LiquidAI/LFM2.5-1.2B-Thinking-ONNX"
MODEL_VARIANTS = {
    "fp16": "model_fp16.onnx",
    "q4": "model_q4.onnx",
    "q8": "model_q8.onnx",
    "base": "model.onnx"
}


def download_onnx_model(
    output_dir: Path,
    variant: str = "base",
    force_download: bool = False
) -> Path:
    """
    Download LFM2.5-Thinking ONNX model from HuggingFace.
    
    Args:
        output_dir: Directory to save the model
        variant: Model variant (base, fp16, q4, q8)
        force_download: Re-download even if file exists
    
    Returns:
        Path to the downloaded ONNX file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = MODEL_VARIANTS.get(variant, MODEL_VARIANTS["base"])
    local_path = output_dir / filename
    
    if local_path.exists() and not force_download:
        print(f"Model already exists at: {local_path}")
        return local_path
    
    print(f"Downloading {MODEL_REPO} ({variant})...")
    print(f"This may take a while depending on your connection speed.")
    
    try:
        # Try to download specific file first
        downloaded_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"Downloaded model to: {downloaded_path}")
        return Path(downloaded_path)
        
    except Exception as e:
        print(f"Could not download specific file: {e}")
        print("Attempting to download full repository...")
        
        # Fall back to downloading the entire repo
        repo_path = snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=output_dir / "repo",
            resume_download=True
        )
        
        # Find ONNX files in downloaded repo
        onnx_files = list(Path(repo_path).glob("*.onnx"))
        if onnx_files:
            print(f"Found ONNX files: {[f.name for f in onnx_files]}")
            return onnx_files[0]
        else:
            raise FileNotFoundError(f"No ONNX files found in {repo_path}")


def download_tokenizer(output_dir: Path) -> Path:
    """Download the tokenizer for LFM2.5-Thinking."""
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    
    # Download tokenizer files
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    
    print("Downloading tokenizer...")
    for filename in tokenizer_files:
        try:
            hf_hub_download(
                repo_id="LiquidAI/LFM2.5-1.2B-Thinking",
                filename=filename,
                local_dir=tokenizer_dir,
                local_dir_use_symlinks=False
            )
        except Exception as e:
            print(f"Warning: Could not download {filename}: {e}")
    
    print(f"Tokenizer saved to: {tokenizer_dir}")
    return tokenizer_dir


def verify_onnx_model(model_path: Path) -> bool:
    """Verify the ONNX model is valid."""
    try:
        import onnx
        print(f"Verifying ONNX model: {model_path}")
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        
        # Print model info
        print(f"  IR Version: {model.ir_version}")
        print(f"  Opset Version: {model.opset_import[0].version}")
        print(f"  Producer: {model.producer_name} {model.producer_version}")
        print(f"  Graph inputs: {len(model.graph.input)}")
        print(f"  Graph outputs: {len(model.graph.output)}")
        print(f"  Graph nodes: {len(model.graph.node)}")
        
        # Check for problematic operators
        op_types = set(node.op_type for node in model.graph.node)
        print(f"  Operator types: {len(op_types)}")
        
        # Flag custom operators that may need decomposition
        custom_ops = [op for op in op_types if op.startswith("LIV") or "Recursive" in op]
        if custom_ops:
            print(f"  ⚠️ Custom operators found: {custom_ops}")
            print("    These may need decomposition for Hailo compatibility.")
        else:
            print("  ✓ No custom operators detected")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and export LFM2.5-Thinking ONNX model"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./models"),
        help="Output directory for the model"
    )
    parser.add_argument(
        "--variant",
        choices=["base", "fp16", "q4", "q8"],
        default="base",
        help="Model variant to download"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists"
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip downloading the tokenizer"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip ONNX model verification"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LFM2.5-1.2B-Thinking ONNX Export")
    print("=" * 60)
    
    # Download model
    model_path = download_onnx_model(
        args.output_dir,
        variant=args.variant,
        force_download=args.force
    )
    
    # Download tokenizer
    if not args.skip_tokenizer:
        download_tokenizer(args.output_dir)
    
    # Verify model
    if not args.skip_verify:
        print("\n" + "-" * 60)
        if verify_onnx_model(model_path):
            print("\n✓ Model export complete and verified!")
        else:
            print("\n⚠️ Model verification had issues. Check logs above.")
    
    print(f"\nModel path: {model_path}")
    return model_path


if __name__ == "__main__":
    main()
