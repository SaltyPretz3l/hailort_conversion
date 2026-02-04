"""
Decompose ONNX operators for Hailo DFC compatibility - v5.

This script rewrites the decomposition to avoid patterns that trigger
Hailo's problematic `is_null_transpose_near_torch_tile` function.

Key changes from v4:
1. Avoid Unsqueeze+Tile+Reshape pattern for KV head expansion
2. Use Split+Concat for head replication (works better with Hailo)
3. Add explicit shape tensors instead of dynamic inference

Usage:
    python decompose_for_hailo.py --input models/lfm2.5_fp16_fixed_v3_static.onnx \
                                  --output models/lfm2.5_fp16_fixed_v6_hailo.onnx
"""

import onnx
from onnx import helper, numpy_helper, TensorProto, shape_inference
import numpy as np
import argparse
from collections import defaultdict, deque
from typing import List, Dict, Tuple


def make_node(op_type: str, inputs: List[str], outputs: List[str], name: str = None, **kwargs):
    """Helper to create ONNX nodes."""
    return helper.make_node(op_type, inputs, outputs, name=name, **kwargs)


def get_attr_value(node, attr_name, default=None):
    """Get attribute value from a node."""
    for attr in node.attribute:
        if attr.name == attr_name:
            if attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            elif attr.type == onnx.AttributeProto.INT:
                return attr.i
            elif attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
    return default


def topological_sort_nodes(nodes: List, original_graph) -> List:
    """Topologically sort ONNX nodes using Kahn's algorithm."""
    node_outputs = {}
    node_inputs = defaultdict(set)
    
    for idx, node in enumerate(nodes):
        for out in node.output:
            if out:
                node_outputs[out] = idx
        for inp in node.input:
            if inp:
                node_inputs[idx].add(inp)
    
    dependents = defaultdict(list)
    in_degree = {i: 0 for i in range(len(nodes))}
    
    for idx, node in enumerate(nodes):
        for inp in node_inputs[idx]:
            if inp in node_outputs:
                producer_idx = node_outputs[inp]
                dependents[producer_idx].append(idx)
                in_degree[idx] += 1
    
    queue = deque()
    for idx in range(len(nodes)):
        if in_degree[idx] == 0:
            queue.append(idx)
    
    sorted_indices = []
    while queue:
        idx = queue.popleft()
        sorted_indices.append(idx)
        
        for dep_idx in dependents[idx]:
            in_degree[dep_idx] -= 1
            if in_degree[dep_idx] == 0:
                queue.append(dep_idx)
    
    if len(sorted_indices) != len(nodes):
        print(f"  ⚠️ Warning: Could not sort all nodes ({len(sorted_indices)}/{len(nodes)})")
        remaining = [i for i in range(len(nodes)) if i not in sorted_indices]
        sorted_indices.extend(remaining)
    
    return [nodes[i] for i in sorted_indices]


class HailoCompatibleDecomposer:
    """Decompose operators with Hailo DFC compatibility in mind."""
    
    def __init__(self, model_path: str):
        print(f"Loading model (lightweight for inference): {model_path}")
        try:
            # 1. Infer shapes on lightweight model (without loading ext data)
            model_light = onnx.load(model_path, load_external_data=False)
            model_light = shape_inference.infer_shapes(model_light)
            print("✓ Lightweight input inference successful")
            
            # 2. Extract value_info
            inferred_value_info = {vi.name: vi for vi in model_light.graph.value_info}
            
            # 3. Load full model
            print("Loading full model...")
            self.model = onnx.load(model_path)
            self.graph = self.model.graph
            
            # 4. Merge inferred info into map (not necessarily into graph value_info list yet, 
            #    but we use the map for lookups)
            
        except Exception as e:
            print(f"⚠️ Warning: Input inference failed: {e}")
            print("Loading full model fallback...")
            self.model = onnx.load(model_path)
            self.graph = self.model.graph
            inferred_value_info = {}

        self.new_nodes = []
        self.new_initializers = []
        
        self.initializers = {init.name: init for init in self.graph.initializer}
        self.new_value_info = []
        
        # Build map of value info for easy lookup
        # Start with existing graph info
        self.value_info_map = {vi.name: vi for vi in self.graph.value_info}
        # Update with inferred info
        self.value_info_map.update(inferred_value_info)
        
        for inp in self.graph.input:
            self.value_info_map[inp.name] = inp
        for out in self.graph.output:
            self.value_info_map[out.name] = out
            
        self.uid_counter = 0
    
    def get_uid(self) -> str:
        self.uid_counter += 1
        return f"_hd_{self.uid_counter}"  # hd = Hailo Decomp
    
    def add_initializer_fp16(self, name: str, value: float) -> str:
        """Add a scalar FP16 initializer."""
        tensor = numpy_helper.from_array(
            np.array(value, dtype=np.float16),
            name=name
        )
        self.new_initializers.append(tensor)
        return name
    
    def add_initializer_int64(self, name: str, values: List[int]) -> str:
        """Add an INT64 initializer."""
        tensor = numpy_helper.from_array(
            np.array(values, dtype=np.int64),
            name=name
        )
        self.new_initializers.append(tensor)
        return name

    def copy_shape(self, input_name: str, output_name: str):
        """Copy shape from input to output value_info."""
        if input_name in self.value_info_map:
            input_vi = self.value_info_map[input_name]
            if input_vi.type.tensor_type.HasField("shape"):
                new_vi = helper.make_tensor_value_info(
                    output_name,
                    input_vi.type.tensor_type.elem_type,
                    None # Shape will be set below
                )
                new_vi.type.tensor_type.shape.CopyFrom(input_vi.type.tensor_type.shape)
                self.new_value_info.append(new_vi)
                self.value_info_map[output_name] = new_vi
                print(f"    + Simple Shape Prop: {input_name} -> {output_name}")

    def decompose_rotary_embedding(self, node) -> List:
        """Decompose RotaryEmbedding into standard ONNX ops."""
        uid = self.get_uid()
        nodes = []
        
        input_tensor = node.input[0]
        position_ids = node.input[1]
        cos_cache = node.input[2]
        sin_cache = node.input[3]
        output_tensor = node.output[0]
        
        # 1. Inspect input shapes to determine dims
        head_dim = 128 # Default fallback
        num_heads = 16 # Default fallback
        hidden_dim = 2048 # Default fallback

        # Try to deduce head_dim from cos_cache [1, 1, 1, head_dim] (usually) or [max_seq, head_dim]
        # In this model, cos_cache seems to be [MaxSeq, HeadDim] or similar.
        # Let's rely on adding a Reshape that forces [B, S, NumHeads, HeadDim] 
        # But we need NumHeads and HeadDim.
        
        # NOTE: LFM 2.5 3B typically has Hidden=2560, Heads=20 -> HeadDim=128
        # OR Hidden=2048, Heads=32 -> HeadDim=64
        # We should try to infer this dynamically if possible, or assume LFM params.
        # Inspecting previous `inspect_rope_shapes.py` output:
        # Input 0 (_hd_1_x2): Shape = [0, 0, 1024] -> This is AFTER split.
        # Split Input Shape: [0, 0, 2048]. So hidden_dim=2048.
        # Unsqueeze Output Shape: [1, 1, 1, 32]. Wait, 32?
        # If cos shape is 32, then head_dim (or half of it) is 32. 
        # RoPE usually applies to a sub-part or full part.
        
        # Let's look at the implementation of RoPE in this model.
        # It splits input into x1, x2 (half heads? or half dim?).
        # Usually RoPE rotates pairs (x, y) -> (x cos - y sin, x sin + y cos).
        # In Llama-style, it rotates the first 50% vs second 50% of head_dim? Or interleaved?
        # The split axis is -1.
        
        # 1. Determine Hidden Dim and Heads Statically (Name-based fallback)
        # Dynamic Shape ops caused DFC parser crash.
        # k_rotary has 8 heads (512 hidden), q_rotary has 32 heads (2048 hidden).
        target_head_dim = 64
        
        if "k_rotary" in node.name:
            num_heads = 8
            print(f"    + Detected k_rotary: Using num_heads={num_heads}")
        else:
            num_heads = 32
            print(f"    + Assumed q_rotary: Using num_heads={num_heads}")

        # 2. Reshape Input to [B, S, Heads, 64] using Explicit Dimensions
        reshaped_input = f"{uid}_reshaped_in"
        shape_const_in = self.add_initializer_int64(f"{uid}_shape_in", [0, -1, num_heads, target_head_dim])
        nodes.append(make_node("Reshape", [input_tensor, shape_const_in], [reshaped_input],
                               name=f"RoPE{uid}_reshape_in"))
        
        # 2. Gather cos/sin (Standard)
        cos_gathered = f"{uid}_cos_gathered"
        sin_gathered = f"{uid}_sin_gathered"
        
        nodes.append(make_node("Gather", [cos_cache, position_ids], [cos_gathered], 
                               name=f"RoPE{uid}_gather_cos", axis=0))
        nodes.append(make_node("Gather", [sin_cache, position_ids], [sin_gathered], 
                               name=f"RoPE{uid}_gather_sin", axis=0))
                               
        # 3. Unsqueeze cos/sin to [1, S, 1, 64/2] -> [1, S, 1, 32]
        # Wait, original gather output is [S, 32] (if cos_cache is [MaxSeq, 32]) or [1, S, 32]?
        # Cos cache usually [MaxSeq, HeadDim/2]. Gather(axis=0) -> [Batch(?), S, HeadDim/2] or just [S, HeadDim/2].
        # If position_ids is [B, S], Gather -> [B, S, HeadDim/2].
        
        # Let's assume position_ids is [B, S].
        # Cos gathered: [B, S, 32].
        # We need it to broadcast to [B, S, Heads, 32].
        # So Unsqueeze axis 2 -> [B, S, 1, 32].
        
        # 3. Unsqueeze cos/sin to [1, S, 1, 64/2] -> [1, S, 1, 32]
        # REPLACE UNSQUEEZE WITH RESHAPE TO AVOID HAILO PARSER ERROR ON RANK 3 INPUTS
        # Target: [B, S, 1, HeadDim]
        # Input: [B, S, HeadDim]
        # Use [0, -1, 1, 32] to infer S, explicit 1 and 32.
        
        reshape_4d_shape = self.add_initializer_int64(f"{uid}_unsq_shape_4d", [0, -1, 1, int(target_head_dim/2)]) 
        cos_unsq = f"{uid}_cos_unsq"
        sin_unsq = f"{uid}_sin_unsq"
        
        nodes.append(make_node("Reshape", [cos_gathered, reshape_4d_shape], [cos_unsq],
                               name=f"RoPE{uid}_reshape_cos"))
        nodes.append(make_node("Reshape", [sin_gathered, reshape_4d_shape], [sin_unsq],
                               name=f"RoPE{uid}_reshape_sin"))
        
        # 4. Split Reshaped Input [B, S, Heads, 64] -> x1, x2 [B, S, Heads, 32]
        # Start using `reshaped_input`
        x1 = f"{uid}_x1"
        x2 = f"{uid}_x2"
        nodes.append(make_node("Split", [reshaped_input], [x1, x2],
                               name=f"RoPE{uid}_split", axis=-1, num_outputs=2))
                               
        # 5. Apply rotation (Broadcasting works now: [B,S,H,32] * [B,S,1,32])
        x1_cos = f"{uid}_x1_cos"
        x2_sin = f"{uid}_x2_sin"
        x1_sin = f"{uid}_x1_sin"
        x2_cos = f"{uid}_x2_cos"
        rot1 = f"{uid}_rot1"
        rot2 = f"{uid}_rot2"
        
        nodes.append(make_node("Mul", [x1, cos_unsq], [x1_cos], name=f"RoPE{uid}_mul_x1_cos"))
        nodes.append(make_node("Mul", [x2, sin_unsq], [x2_sin], name=f"RoPE{uid}_mul_x2_sin"))
        nodes.append(make_node("Mul", [x1, sin_unsq], [x1_sin], name=f"RoPE{uid}_mul_x1_sin"))
        nodes.append(make_node("Mul", [x2, cos_unsq], [x2_cos], name=f"RoPE{uid}_mul_x2_cos"))
        
        nodes.append(make_node("Sub", [x1_cos, x2_sin], [rot1], name=f"RoPE{uid}_sub"))
        nodes.append(make_node("Add", [x1_sin, x2_cos], [rot2], name=f"RoPE{uid}_add"))
        
        # 6. Concat -> [B, S, Heads, 64]
        # Output DIRECTLY to output_tensor (Rank 4).
        # Do NOT flatten back to Rank 3.
        nodes.append(make_node("Concat", [rot1, rot2], [output_tensor],
                               name=f"RoPE{uid}_concat", axis=-1))
                               
        # Note: We do NOT call copy_shape because the shape changes from Rank 3 to Rank 4.
        # We rely on shape inference to update the value_info for output_tensor.
        
        return nodes

    def decompose_skip_simplified_layernorm(self, node) -> List:
        """Decompose SkipSimplifiedLayerNormalization into Add + RMSNorm."""
        uid = self.get_uid()
        nodes = []
        
        input_tensor = node.input[0]
        skip_tensor = node.input[1]
        weight = node.input[2] if len(node.input) > 2 else None
        output_tensor = node.output[0]
        
        epsilon = get_attr_value(node, 'epsilon', 1e-5)
        
        if input_tensor == skip_tensor:
            added = input_tensor
        else:
            added = f"{uid}_added"
            nodes.append(make_node("Add", [input_tensor, skip_tensor], [added],
                                   name=f"SkipLN{uid}_add"))
        
        # RMSNorm: y = x * w / sqrt(mean(x^2) + eps)
        sq = f"{uid}_sq"
        nodes.append(make_node("Mul", [added, added], [sq], name=f"SkipLN{uid}_sq"))
        
        reduce_axes = self.add_initializer_int64(f"{uid}_axes", [-1])
        mean = f"{uid}_mean"
        nodes.append(make_node("ReduceMean", [sq, reduce_axes], [mean],
                               name=f"SkipLN{uid}_mean", keepdims=1))
        
        eps_name = self.add_initializer_fp16(f"{uid}_eps", epsilon)
        mean_eps = f"{uid}_mean_eps"
        nodes.append(make_node("Add", [mean, eps_name], [mean_eps], name=f"SkipLN{uid}_add_eps"))
        
        denom = f"{uid}_denom"
        nodes.append(make_node("Sqrt", [mean_eps], [denom], name=f"SkipLN{uid}_sqrt"))
        
        rsqrt = f"{uid}_rsqrt"
        nodes.append(make_node("Reciprocal", [denom], [rsqrt], name=f"SkipLN{uid}_recip"))
        
        normed = f"{uid}_normed"
        nodes.append(make_node("Mul", [added, rsqrt], [normed], name=f"SkipLN{uid}_norm"))
        
        if weight:
            nodes.append(make_node("Mul", [normed, weight], [output_tensor],
                                   name=f"SkipLN{uid}_scale"))
        else:
            nodes.append(make_node("Identity", [normed], [output_tensor],
                                   name=f"SkipLN{uid}_identity"))
        
        # LayerNorm preserves shape (mostly - if reducing last dim, but here it keeps dims or broadcasts? 
        # Wait, SkipSimplifiedLayerNormalization Usually preserves shape of input[0])
        self.copy_shape(input_tensor, output_tensor)
        
        return nodes

    def decompose_group_query_attention(self, node) -> List:
        """
        Decompose GroupQueryAttention using Hailo-compatible patterns.
        
        KEY CHANGE: Instead of Unsqueeze+Tile+Reshape for head expansion,
        we use Split to separate heads, then Concat to repeat them.
        This avoids the Transpose-near-Tile pattern that crashes Hailo.
        """
        uid = self.get_uid()
        nodes = []
        
        q = node.input[0]
        k = node.input[1]
        v = node.input[2]
        past_key = node.input[3]
        past_value = node.input[4]
        
        output = node.output[0]
        present_key = node.output[1] if len(node.output) > 1 else None
        present_value = node.output[2] if len(node.output) > 2 else None
        
        num_heads = get_attr_value(node, 'num_heads', 32)
        kv_num_heads = get_attr_value(node, 'kv_num_heads', 8)
        scale = get_attr_value(node, 'scale', 0.125)
        
        head_dim = 64
        hidden_dim = num_heads * head_dim  # 2048
        groups = num_heads // kv_num_heads  # 4
        
        # ===== Transpose K, V: [B, S, kv_heads, head_dim] -> [B, kv_heads, S, head_dim] =====
        k_t = f"{uid}_k_t"
        v_t = f"{uid}_v_t"
        
        # KEY CHANGE: Check if inputs are Rank 3 [B, S, H*D]. If so, Reshape to [B, S, H, D]
        # This fixes IndexError if upstream ops (RoPE) produce flattened outputs.
        
        def ensure_rank4(tensor_name, n_heads, h_dim):
            should_reshape = False
            
            if tensor_name in self.value_info_map:
                vi = self.value_info_map[tensor_name]
                # If we know the shape and it's NOT Rank 4, we must reshape.
                # (e.g. Rank 3 [B, S, Hidden], or Rank 2 [Tokens, Hidden])
                if len(vi.type.tensor_type.shape.dim) != 4:
                     should_reshape = True
            else:
                # If unknown, assume it needs reshaping (safest for v_proj outputs)
                should_reshape = True
            
            if should_reshape:
                 # Reshape [B, S, Hidden] -> [B, S, Heads, HeadDim]
                 # Use [0, -1, ...] to be safe with Hailo parser (avoiding 0 for inferred dim)
                 reshaped = f"{uid}_reshaped_{tensor_name}"
                 shape_const = self.add_initializer_int64(f"{uid}_shape_{tensor_name}", [0, -1, n_heads, h_dim])
                 nodes.append(make_node("Reshape", [tensor_name, shape_const], [reshaped],
                                        name=f"GQA{uid}_reshape_in_{tensor_name}"))
                 
                 # Do NOT copy shape! Reshape changes it.
                 # Rely on infer_shapes to calculate [B, S, H, D]
                 return reshaped
            
            return tensor_name

        q = ensure_rank4(q, num_heads, head_dim)
        k = ensure_rank4(k, kv_num_heads, head_dim)
        v = ensure_rank4(v, kv_num_heads, head_dim)
        
        # Now Transpose using potentially reshaped inputs
        nodes.append(make_node("Transpose", [k], [k_t],
                               name=f"GQA{uid}_tr_k", perm=[0, 2, 1, 3]))
        nodes.append(make_node("Transpose", [v], [v_t],
                               name=f"GQA{uid}_tr_v", perm=[0, 2, 1, 3]))
        
        # ===== Concat with past cache =====
        k_cat = f"{uid}_k_cat"
        v_cat = f"{uid}_v_cat"
        nodes.append(make_node("Concat", [past_key, k_t], [k_cat],
                               name=f"GQA{uid}_cat_k", axis=2))
        nodes.append(make_node("Concat", [past_value, v_t], [v_cat],
                               name=f"GQA{uid}_cat_v", axis=2))
        
        # Present KV for next iteration
        if present_key:
            nodes.append(make_node("Identity", [k_cat], [present_key],
                                   name=f"GQA{uid}_present_k"))
        if present_value:
            nodes.append(make_node("Identity", [v_cat], [present_value],
                                   name=f"GQA{uid}_present_v"))
        
        # ===== Repeat KV heads using Split + Concat (Hailo-friendly) =====
        # Instead of Tile, we split by heads and repeat via Concat
        # k_cat: [B, kv_heads, seq, head_dim] -> need [B, num_heads, seq, head_dim]
        # Each of the kv_heads needs to be repeated 'groups' times
        
        # Split along head dimension (axis 1)
        k_head_outputs = [f"{uid}_k_head_{i}" for i in range(kv_num_heads)]
        v_head_outputs = [f"{uid}_v_head_{i}" for i in range(kv_num_heads)]
        
        # Split K into individual heads
        split_sizes_k = self.add_initializer_int64(f"{uid}_split_sizes", [1] * kv_num_heads)
        nodes.append(make_node("Split", [k_cat, split_sizes_k], k_head_outputs,
                               name=f"GQA{uid}_split_k", axis=1))
        
        # Split V into individual heads
        split_sizes_v = self.add_initializer_int64(f"{uid}_split_sizes_v", [1] * kv_num_heads)
        nodes.append(make_node("Split", [v_cat, split_sizes_v], v_head_outputs,
                               name=f"GQA{uid}_split_v", axis=1))
        
        # Repeat each head 'groups' times by concatenating
        k_repeated_heads = []
        v_repeated_heads = []
        for i in range(kv_num_heads):
            # k_head_i: [B, 1, seq, head_dim] - repeat 'groups' times
            for g in range(groups):
                k_repeated_heads.append(k_head_outputs[i])
                v_repeated_heads.append(v_head_outputs[i])
        
        # Concat all repeated heads: -> [B, num_heads, seq, head_dim]
        k_rep = f"{uid}_k_rep"
        v_rep = f"{uid}_v_rep"
        nodes.append(make_node("Concat", k_repeated_heads, [k_rep],
                               name=f"GQA{uid}_rep_k", axis=1))
        nodes.append(make_node("Concat", v_repeated_heads, [v_rep],
                               name=f"GQA{uid}_rep_v", axis=1))
        
        # ===== Scaled Dot-Product Attention =====
        # Q: [B, S, num_heads, head_dim] -> [B, num_heads, S, head_dim]
        q_t = f"{uid}_q_t"
        nodes.append(make_node("Transpose", [q], [q_t],
                               name=f"GQA{uid}_tr_q", perm=[0, 2, 1, 3]))
        
        # K^T: [B, num_heads, seq, head_dim] -> [B, num_heads, head_dim, seq]
        k_t_for_mm = f"{uid}_k_t_mm"
        nodes.append(make_node("Transpose", [k_rep], [k_t_for_mm],
                               name=f"GQA{uid}_tr_k_mm", perm=[0, 1, 3, 2]))
        
        # scores = Q @ K^T
        scores = f"{uid}_scores"
        nodes.append(make_node("MatMul", [q_t, k_t_for_mm], [scores],
                               name=f"GQA{uid}_qk"))
        
        # Scale
        scale_val = self.add_initializer_fp16(f"{uid}_scale", scale)
        scores_scaled = f"{uid}_scores_s"
        nodes.append(make_node("Mul", [scores, scale_val], [scores_scaled],
                               name=f"GQA{uid}_scale"))
        
        # Softmax
        attn_weights = f"{uid}_attn_w"
        nodes.append(make_node("Softmax", [scores_scaled], [attn_weights],
                               name=f"GQA{uid}_softmax", axis=-1))
        
        # output = weights @ V
        attn_out = f"{uid}_attn_out"
        nodes.append(make_node("MatMul", [attn_weights, v_rep], [attn_out],
                               name=f"GQA{uid}_av"))
        
        # Transpose back: [B, num_heads, S, head_dim] -> [B, S, num_heads, head_dim]
        attn_tr = f"{uid}_attn_tr"
        nodes.append(make_node("Transpose", [attn_out], [attn_tr],
                               name=f"GQA{uid}_tr_out", perm=[0, 2, 1, 3]))
        
        # Reshape to [B, S, hidden_dim]
        out_shape = self.add_initializer_int64(f"{uid}_out_shape", [1, -1, hidden_dim])
        nodes.append(make_node("Reshape", [attn_tr, out_shape], [output],
                               name=f"GQA{uid}_reshape"))
        
        return nodes

    def decompose_all(self):
        """Process all nodes and decompose unsupported operators."""
        original_nodes = list(self.graph.node)
        final_nodes = []
        
        stats = {"RotaryEmbedding": 0, "SkipSimplifiedLayerNormalization": 0, "GroupQueryAttention": 0}
        
        print(f"\nProcessing {len(original_nodes)} nodes...")
        
        for node in original_nodes:
            if node.op_type == "RotaryEmbedding":
                decomposed = self.decompose_rotary_embedding(node)
                final_nodes.extend(decomposed)
                stats["RotaryEmbedding"] += 1
                print(f"  ✓ RotaryEmbedding: {node.name} -> {len(decomposed)} nodes")
                
            elif node.op_type == "SkipSimplifiedLayerNormalization":
                decomposed = self.decompose_skip_simplified_layernorm(node)
                final_nodes.extend(decomposed)
                stats["SkipSimplifiedLayerNormalization"] += 1
                print(f"  ✓ SkipSimplifiedLayerNormalization: {node.name} -> {len(decomposed)} nodes")
                
            elif node.op_type == "GroupQueryAttention":
                decomposed = self.decompose_group_query_attention(node)
                final_nodes.extend(decomposed)
                stats["GroupQueryAttention"] += 1
                print(f"  ✓ GroupQueryAttention: {node.name} -> {len(decomposed)} nodes")
                
            else:
                final_nodes.append(node)
        
        print(f"\nDecomposition summary:")
        for op, count in stats.items():
            print(f"  {op}: {count}")
        
        print(f"\nTotal: {len(original_nodes)} -> {len(final_nodes)} nodes")
        
        # Topological sort
        print("\nTopologically sorting...")
        sorted_nodes = topological_sort_nodes(final_nodes, self.graph)
        
        return sorted_nodes
    
    def save(self, output_path: str, sorted_nodes: List):
        """Save the modified model."""
        new_graph = helper.make_graph(
            sorted_nodes,
            self.graph.name,
            self.graph.input,
            self.graph.output,
            list(self.graph.initializer) + self.new_initializers
        )
        
        new_graph.value_info.extend(self.graph.value_info)
        new_graph.value_info.extend(self.new_value_info)
        
        new_model = helper.make_model(new_graph, opset_imports=self.model.opset_import)
        
        print(f"\nSaving to {output_path}...")
        onnx.save(
            new_model,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{output_path.split('/')[-1].split(chr(92))[-1]}.data",
            size_threshold=1024,
            convert_attribute=False
        )
        print("✓ Model saved!")
        
        print("Running shape inference on saved model...")
        try:
            shape_inference.infer_shapes_path(output_path, output_path)
            print("✓ Shape inference successful")
        except Exception as e:
            print(f"⚠️ Warning: Shape inference failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Decompose ONNX operators for Hailo compatibility")
    parser.add_argument("--input", required=True, help="Input ONNX model path")
    parser.add_argument("--output", required=True, help="Output ONNX model path")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ONNX Decomposition for Hailo DFC (v6)")
    print("=" * 60)
    
    decomposer = HailoCompatibleDecomposer(args.input)
    sorted_nodes = decomposer.decompose_all()
    decomposer.save(args.output, sorted_nodes)
    
    # Verify
    print("\nVerifying...")
    try:
        model = onnx.load(args.output, load_external_data=False)
        
        unsupported = set()
        for node in model.graph.node:
            if node.op_type in {"RotaryEmbedding", "SkipSimplifiedLayerNormalization", "GroupQueryAttention"}:
                unsupported.add(node.op_type)
        
        if unsupported:
            print(f"⚠️ Still found: {unsupported}")
        else:
            print("✓ No unsupported operators!")
            
        # Check for Tile (should be none)
        tile_count = sum(1 for n in model.graph.node if n.op_type == "Tile")
        if tile_count > 0:
            print(f"⚠️ Found {tile_count} Tile nodes")
        else:
            print("✓ No Tile nodes!")
            
        print(f"✓ Model has {len(model.graph.node)} nodes")
    except Exception as e:
        print(f"⚠️ Verification: {e}")
    
    print("\n" + "=" * 60)
    print("Done! Test with: python3 test_v6.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
