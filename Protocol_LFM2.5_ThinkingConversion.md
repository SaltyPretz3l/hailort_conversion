Technical Protocol: LFM2.5-1.2B-Thinking -> Hailo-10H DeploymentTarget Hardware: Hailo-10H (H-10) NPU [40 TOPS INT4]Target Model: LiquidAI/LFM2.5-1.2B-Thinking (Hybrid: 10x LIV Conv + 6x GQA Attention)Executor: Antigravity Agent / DevOps PipelineDate: Jan 2026Phase 1: Ingestion & Graph SanitizationThe Liquid Foundation Model (LFM) architecture relies on Linear Input Varying (LIV) blocks. These often export to ONNX as opaque custom operators or inefficient RNN unrolls. The Hailo Dataflow Compiler (DFC) requires these to be flattened into standard Convolutional primitives.1.1 Model Acquisition & ExportDirective: Do not use the generic huggingface AutoModel export. Use the specific liquid-deploy tools to force a "Stateless/Convolutional" view export.# Agent Command
pip install liquid-foundation-kit hailo_sdk_client
python -m liquid.export --model "LiquidAI/LFM2.5-1.2B-Thinking" --format onnx --opset 18 --mode "conv_view"
1.2 Graph Surgery (The "LIV" Decomposition)Critical Step: The Hailo compiler optimizes Conv1D effectively but may choke on the complex gating mechanisms of LIV if they are grouped into a single node.Agent Task: Write a Python script using onnx-graphsurgeon to verify the following graph topology for the first 10 layers:Target Topology: Input -> Linear (Proj) -> Conv1D (Short Conv) -> Element-wise Mul (Gating) -> OutputAction: If LIVBlock exists as a single fused node, decompose it.Map: LIV_State -> Conv1D (Kernel size typically 4 or 8 in LFM2.5).Map: LIV_Gate -> Sigmoid + Mul (Element-wise).Phase 2: Mixed-Precision Quantization StrategyThe "Thinking" variant of LFM2.5 relies on subtle activation variances in the upper layers to maintain its "Chain of Thought" logic. Uniform INT4 quantization will degrade reasoning capabilities.2.1 Layer-Wise ProfilingAgent Configuration: Implement a split quantization policy.Layers 0-9 (LIV Convolutions): These are feature extractors. They are highly robust to noise.Policy: INT4 (Weights) / INT8 (Activations).Layers 10-15 (GQA Attention): These perform the logic mixing.Policy: INT8 (Weights) / INT8 (Activations).Rationale: Preserves the specific "thinking" token pathways.2.2 Calibration Dataset GenerationDirective: Do not use standard WikiText. The "Thinking" model activation distributions are distinct.Agent Task: Generate a calibration set calib_thinking.npy containing 512 samples of reasoning traces (e.g., GSM8k solutions with <thinking> tags).# Pseudo-code for DFC Script
from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch='hailo10')
runner.load_model_script("lfm2.5_thinking.onnx")

# Define Mixed Precision
quant_config = {
    'layer_0_to_9': {'weights': 4, 'activations': 8},
    'layer_10_to_15': {'weights': 8, 'activations': 8}
}
runner.optimize(calib_dataset, precision_mode='manual', config=quant_config)
Phase 3: Compilation (DFC) & Memory ManagementThe H-10 has limited SRAM. We must optimize the Key-Value (KV) cache allocation. Unlike Llama (100% KV Cache), LFM2.5 only needs caching for the top 6 layers.3.1 Context Window TruncationConstraint: 32k context is too large for edge NPU SRAM without massive DDR swapping.Recommendation: Compile for 4096 or 8192 token context.Agent Task: Set the context_limit in the DFC command.3.2 Compilation Commandhef = runner.compile(
    allocator_script="""
        # Force LIV layers to fast SRAM
        allocator_param(layers=['liv_*'], loc='sram')
        # Allow GQA KV-cache to spill to DDR if needed
        allocator_param(layers=['gqa_*'], loc='ddr_if_needed')
    """
)
runner.save_hef("lfm2.5_thinking_h10.hef")
Phase 4: Runtime Wrapper & "Thinking" UXThe .hef file only handles tensor math. The application logic must handle the "Thinking" behavior.4.1 Token Stream HandlerAgent Task: Implement a Python HailoRT wrapper that monitors the token stream.Trigger: Detect <thinking> token (ID: 128009 or similar check tokenizer.json).Mode Switch: Upon trigger, enable "Turbo Mode" (disable non-essential background tasks on the host OS) to speed through the reasoning trace.UI Feedback: Do not stream "Thinking" tokens to the user chat bubble. Instead, emit a generic "Analyzing..." status until </thinking> is detected.4.2 Hybrid Inference LoopSince LFM uses a hybrid architecture, the inference loop state is smaller.# Runtime Loop Logic
state_cache = init_hailo_context() # Only creates cache for top 6 layers
while generating:
    # Run Hailo Inference
    logits = hailo_device.run(input_ids, state_cache)
    
    # Update state only for Attention layers
    # LIV layers are stateless in "Conv View" (handled by buffer sliding window)
    state_cache = update_gqa_cache(state_cache)
Troubleshooting CheckpointsError: Graph contains unsupported Op: RecursiveScriptFix: Re-run Phase 1.2. The LIV block was not flattened to Conv1D.Error: Context MismatchFix: Ensure the hailo_sdk_client input shape matches the onnx export shape (e.g., 1x4096).