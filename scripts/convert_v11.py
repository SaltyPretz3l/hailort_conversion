
import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    base_model = "models/lfm2.5_fp16_fixed_v3_static.onnx"
    step1_model = "models/lfm2.5_fp16_fixed_v11_step1.onnx"
    final_model = "models/lfm2.5_fp16_fixed_v11_hailo.onnx"
    
    if not os.path.exists(base_model):
        print(f"Error: Base model {base_model} not found.")
        return

    # 1. Decompose (Applies RoPE fix via modified decompose_for_hailo.py)
    print("\n" + "="*50)
    print("Step 1: Decomposition (v6 logic + RoPE fix)")
    print("="*50)
    run_cmd(f"python scripts/decompose_for_hailo.py --input {base_model} --output {step1_model}")
    
    # 2. Promote Init to Input (v10 fix)
    print("\n" + "="*50)
    print("Step 2: Promoting Embeddings to Input")
    print("="*50)
    target_name = "model.embed_tokens.weight"
    run_cmd(f"python scripts/promote_init_to_input.py {step1_model} {final_model} {target_name}")
    
    print("\n" + "="*50)
    print(f"✓ v11 Model Generated: {final_model}")
    print("="*50)

if __name__ == "__main__":
    main()
