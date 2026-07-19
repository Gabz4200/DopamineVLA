# VELA Post-Mortem: What to Keep, What to Drop

Analysis of `VELA/` for potential reuse in DopamineVLA or other projects.

---

## What's Good (Worth Reusing)

### 1. WindBackstepping CUDA Kernel
**File**: `VELA-v7/src/models/vlm.py:38-285`, `VELA-v7/cuda/wkv7_cuda.cu`

Custom WKV-7 recurrence CUDA kernel with full autograd support (forward + backward). Includes:
- `wind_backstepping_ref_forward/backward` — pure PyTorch reference implementation
- `WindBackstepping` — `torch.autograd.Function` dispatching to C++ extension when available
- GPU detection across CUDA, ROCm, and Metal
- Automatic compilation with architecture-specific flags

**Verdict**: Standalone, reusable. If any project needs WKV-7 on GPU, this is ready to go.

### 2. Block Attention Residuals (Block AttnRes)
**File**: `VELA-v7/src/models/vlm.py:469-474, 477-518`

The `block_attn_res` function and its integration in `Block`:
```python
def block_attn_res(V_blocks, partial_block, proj, norm):
    V = torch.cat([V_blocks, partial_block.unsqueeze(0)], dim=0)
    K = norm(V)
    logits = torch.einsum("d, n b t d -> n b t", proj.weight.squeeze(0), K)
    h = torch.einsum("n b t, n b t d -> b t d", logits.softmax(0), V)
    return h
```

Partitions layers into chunks, accumulates hidden states in `V_blocks`, and uses learned cross-layer attention to selectively aggregate previous representations. Solves the PreNorm hidden-state dilution problem.

**Verdict**: Elegant, self-contained concept (~10 lines). Could be adapted for DopamineVLA's action head if it ever needs multi-layer feature aggregation.

### 3. MHC MoE (Manifold-Constrained Hyper-Connections)
**File**: `VELA-v7/src/models/vlm.py:521-620`

`MHCBlock` replaces standard FFN layers in the first 4 blocks with 4-expert MoE using:
- Sinkhorn-Knopp doubly stochastic routing (20 iterations)
- Hyper-connection parameters (`phi`, `alpha`, `b`) for pre/post/residual gating
- RMSNorm-based router input

**Verdict**: Well-implemented but overkill for DopamineVLA. Useful if you ever need MoE in an RWKV-based model.

### 4. SigLino Integration Pattern
**File**: `VELA-v7/src/siglino/` (entire directory), `VELA-v7/src/models/vlm.py:765-951`

Locally vendored SigLino vision encoder with:
- Device-aware attention dispatch (flex_attention on CUDA, compiled SDPA on CPU)
- `load_siglino_from_hub()` — loads from HuggingFace without `trust_remote_code`
- `MLPWithContextGating` — gated projection from vision dim to model dim
- `VisualTokenCompressor` — bidirectional RWKV compressor with alternating flip

**Verdict**: DopamineVLA already has its own SigLino. The `MLPWithContextGating` projection and `VisualTokenCompressor` are the reusable pieces if you ever need visual token reduction.

### 5. Early Visual Fusion Pattern
**File**: `VELA-v7/src/models/vlm.py:968-995`

The `preparing_embedding` method injects visual tokens in-place:
```python
input_embeds[selected] = image_features  # fill <image> token positions
```

**Verdict**: Simple, clean pattern. DopamineVLA already does this via `DopamineVLAInputsMerger` (Perceiver latents replace `<image>` tokens). Same idea, different mechanism.

### 6. FlowMatchingHead
**File**: `VELA-v7/src/models/vla.py:81-152`

NitroGen-style DiT with:
- Sinusoidal timestep encoding + MLP
- AdaLayerNorm for timestep conditioning
- Cross-attention on Attention Residuals from all backbone layers
- Beta(1.5, 1.0) timestep sampling
- Euler-step ODE solving for inference (distilled to 2 steps)

**Verdict**: Clean implementation. The DiT blocks are standard, but the cross-attention conditioning on `V_blocks` is specific to VELA's architecture. DopamineVLA's action head (Time2Vec + SWA) is a different approach.

### 7. Dataset Handling
**File**: `VELA-v7/src/dataset.py`

Multi-image collation, ChatML formatting with dynamic target masking, image token processing with `<img_start>`/`<img_end>` wrapping.

**Verdict**: Reusable if you need multi-image VLM training data pipelines.

---

## What's Bad (Drop These)

### 1. Multi-Scale Vision Backbone (SAM + DINOv2 + SigLino)
Three vision encoders fused together is overengineered. No evidence that combining all three beats just using SigLino alone. The feature fusion adds complexity without proven benefit.

### 2. The Full VLA Architecture
The dual-head design (InfoNCE world model + flow matching motor controller) is conceptually interesting but completely unvalidated. No training results, no benchmarks, no comparison to baselines. It's a design sketch, not a working system.

### 3. Codebase Sprawl
1037-line `vlm.py` containing: WindBackstepping kernel wrapper, RWKV-7 time/channel mixing, Block, MHCBlock, VisualTokenCompressor, MLPWithContextGating, VLM, dataset embedding logic, and generation. This should be 5-6 separate files.

### 4. DeepSpeed + PyTorch Lightning Coupling
The training infrastructure is tightly coupled to DeepSpeed ZeRO + Lightning. This makes it hard to test components in isolation or run lightweight experiments. DopamineVLA's pure PyTorch approach is more portable.

### 5. Zero Empirical Validation
The entire project has no training results. The README describes architecture but provides no numbers. Without evidence that the design choices actually work, the code is speculative.

---

## Summary: What to Extract

| Component | Reusable? | Where to Put It |
|---|---|---|
| WindBackstepping CUDA kernel | Yes, standalone | `cuda/` directory, any RWKV-7 project |
| Block AttnRes | Yes, ~10 lines | Could enhance DopamineVLA's action head |
| MHC MoE | Maybe, if needed | Only for RWKV MoE experiments |
| SigLino vendored code | Already in DopamineVLA | Skip |
| MLPWithContextGating | Small, useful | Vision-to-LM projection |
| VisualTokenCompressor | If needed | Only if visual token count is a problem |
| FlowMatchingHead | Different approach | DopamineVLA's action head is already better suited |
| Dataset pipeline | Reusable | Multi-image VLM training |
| ChatML formatting | Reusable | Any chat-style VLM training |
| Everything else | Drop | - |

**Bottom line**: The most portable piece is the **WindBackstepping CUDA kernel**. Everything else is either already in DopamineVLA, architecture-specific to VELA's design choices, or unvalidated. The Block AttnRes concept is elegant enough to keep in mind but doesn't need immediate extraction.
