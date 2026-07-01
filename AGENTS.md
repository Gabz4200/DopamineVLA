# Repository Guidelines

## Project Overview

**DopamineVLA** — real-time Reinforcement Learning with explicit decay on Vision-Language-Action models. A self-contained vision-language-action model that depends **only on PyTorch and base transformers** (`PreTrainedConfig`, `PreTrainedModel`, `ModelOutput`, `GenerationMixin`). No SmolVLM, Idefics, or other HF model subclass dependencies.

Two sub-packages:
- `dopaminevla.models.dopamine` — the VLA model (vision encoder + connector + text decoder + action head)
- `dopaminevla.models.siglino` — standalone vision encoder (SigLino, originally Falcon Vision from TII), independently usable

## Architecture & Data Flow

```
pixel_values → SigLinoHFModel → patch_features
    → DopamineVLAConnector (Perceiver cross-attn) → fixed N latents
    → DopamineVLAInputsMerger (replaces <image> tokens with latents)
    → Text Decoder (HF AutoModel, default Llama)
        → lm_head → logits
        → ActionHead (all hidden states)
            → Time2Vec → Cross-attention → SWA → tanh(delta)
            → Persistent state accumulator (fade * prev + delta)
```

### Key modules

| Module | File | Role |
|---|---|---|
| `DopamineVLAConfig` | `configuration_dopaminevla.py` | Top-level config: vision, text, connector, action params |
| `DopamineVLAVisionConfig` | `configuration_dopaminevla.py` | Vision wrapper config (subclass of `SigLinoConfig`) |
| `DopamineVLAPreTrainedModel` | `base.py` | HF `PreTrainedModel` base with flags |
| `DopamineVLAVisionTransformer` | `vision.py` | Single-pass SigLino encoder wrapper |
| `DopamineVLAConnector` | `connector.py` | Perceiver: cross-attn learns N latents from variable-length patch features |
| `DopamineVLAInputsMerger` | `merger.py` | Replaces `<image>` token embeddings with Perceiver latents |
| `DopamineVLAModel` | `model.py` | Core model: vision + connector + text decoder |
| `DopamineVLAForConditionalGeneration` | `generation.py` | LM head + ActionHead + `GenerationMixin` |
| `ActionHead` | `action.py` | Action prediction: Time2Vec → cross-attn → SWA → delta projection |
| `ActionSWABlock` | `action.py` | Sliding-window attention block with KV cache |
| `Outputs` | `outputs.py` | Three dataclasses for base/CausalLM/action outputs |

### SigLino vision encoder (`dopaminevla.models.siglino.siglino`)

| Module | File | Role |
|---|---|---|
| `SigLino` | `model.py` | Main vision encoder: Conv2d patch embed + transformer + teacher adapters |
| `Attention` | `attention.py` | MHA with RoPE + QK-norm + sliding-window/block-sparse attention |
| `TransformerBlock` | `model.py` | Single block: attention + FFN/MoE layer |
| `MoE` / `FeedForward` | `moe.py` | Mixture-of-Experts or dense FFN |
| `SigLinoConfig` / `SigLinoHFModel` | `hf_integration.py` | HF integration classes |
| `SigLinoImageProcessor` | `image_processor.py` | Image preprocessing (smart resize, normalize) |

## Key Directories

```
dopaminevla/
  __init__.py              # Package root, re-exports models
  models/
    __init__.py             # Re-exports dopamine, siglino
    dopamine/               # VLA model
      11 source files
    siglino/                # Vision encoder
      siglino/              # 9 source files
      pca_maps.py           # PCA visualization script
tests/                      # pytest tests
  test_dopamine.py          # VLA model tests (46 tests)
  test_siglino.py           # Vision encoder tests
  test_pca_maps.py          # PCA tests
```

## Development Commands

```bash
# Lint & format
ruff check --fix
ruff format

# Type checking
pyrefly check

# Test
uv run pytest -x -q                                       # fast, stop on first failure
uv run pytest tests/test_dopamine.py -x -q                 # single test file
uv run pytest -m "not slow" -x -q                          # skip slow tests

# Run (inference demo)
uv run dopaminevla/models/dopamine/inference.py --help

# PCA visualization
uv run dopaminevla/models/siglino/pca_maps.py --config_name dense-0.6B --device cpu
```

## Code Conventions & Common Patterns

### Formatting

- **Ruff**: line length 100, target Python 3.13. Lint: `E`, `F`, `W`, `I`. Format with `ruff format`.
- **Type checker**: `pyrefly` with strict settings (`implicit-any = false`, `unannotated-return = false`, `unannotated-parameter = false`). Test files have relaxed `unannotated-return = true`.

### Naming

- Classes: `PascalCase`, prefixed with project (`DopamineVLA*`, `SigLino*`)
- Functions/methods: `snake_case`
- Private attributes: `_leading_underscore` (e.g., `_step_counter`, `_action_buffer`, `_swa_kv_caches`)
- Config params: `snake_case`
- PyTorch module conventions: `forward()`, `post_init()`, `init_weights()`

### Error handling

- **No try/except for control flow.** Only handle truly exceptional external failures (network, file I/O). Use `try/except` for cross-platform imports only.
- **Fail fast** — let developer bugs crash with clean tracebacks. Use `assert` for invariant checks, not validation.
- Guard clauses with `raise ValueError(...)` for invalid inputs.

### Device agnosticism

- All code must work on CPU and CUDA. Derive device from input tensors, never hardcode `"cuda"`.
- No `torch.cuda.is_available()` branching in model logic.
- Use `self.dtype` / `self.device` properties where needed.

```python
# Correct
device = pixel_values.device
x = x.to(dtype=self.dtype, device=inputs_embeds.device)
```

### Imports & dependencies

- **No `transformers.models.*` subclasses** — only `PreTrainedModel`, `PreTrainedConfig`, `ModelOutput`, `GenerationMixin`.
- SigLino was extracted from SmolVLM into self-contained code. No leftover SmolVLM imports.
- Prefer PyTorch native ops (`F.scaled_dot_product_attention`, `F.rms_norm`) over custom implementations where possible.

### State management

- Action head uses persistent state: `_action_buffer` (register buffer), `persistent_action_state`, `_step_counter`.
- Call `action_head.reset_state()` before new episodes.
- SWA layers maintain optional KV caches (`_swa_kv_caches`) for incremental generation.
- `action_head.forward(..., use_cache=True)` enables cache mode.

### Weight initialization

- Custom `_init_weights()` on models with non-default init strategies (ActionHead: truncated normal + xavier with gains).
- Teacher components (SigLino adapters) frozen after init: `param.requires_grad = False`.

## Important Files

| File | Significance |
|---|---|
| `dopaminevla/models/dopamine/__init__.py` | Public API surface — all 15 exported symbols |
| `dopaminevla/models/dopamine/configuration_dopaminevla.py` | Config with ~30 params for vision/text/connector/action |
| `dopaminevla/models/dopamine/model.py` | Core forward pass: vision → connector → text |
| `dopaminevla/models/dopamine/generation.py` | `prepare_inputs_for_generation` logic (KV cache, image encoding lifecycle) |
| `dopaminevla/models/dopamine/action.py` | Action head with sliding-window attention and persistent state |
| `dopaminevla/models/siglino/siglino/model.py` | SigLino vision encoder (patch embed + transformer + teacher adapters) |
| `pyproject.toml` | Project metadata, ruff config, pytest config, uv sources |
| `pyrefly.toml` | Type checking strictness levels |

## Runtime/Tooling Preferences

- **Python**: 3.12 (`>=3.11,<3.14`)
- **Package manager**: `uv` (not pip/poetry). Lockfile: `uv.lock`.
- **PyTorch**: Nightly CPU index from `download.pytorch.org/whl/nightly/cpu` for CI; GPU builds use CUDA index on hardware.
- **No pre-commit hooks** — run `ruff check --fix && ruff format && pyrefly check && uv run pytest` before committing.
- **Linux only** (`sys_platform == 'linux' and platform_machine == 'x86_64'`).

## Testing & QA

- **Framework**: pytest with no plugins.
- **Location**: `tests/test_dopamine.py`, `tests/test_siglino.py`, `tests/test_pca_maps.py`
- **Running**: `uv run pytest -x -q` (stop on first failure, quiet mode).
- **Markers**: `slow` for tests that download models or run large inference.
- **Style**: class-based test organization (`TestConfig`, `TestActionHead`, etc.).
- **What's tested**: shape correctness, NaN absence, causality (action SWA), KV cache parity, gradient flow, weight initialization, serialization round-trip, action head state reset, edge cases (zero-image filtering, mutual exclusion of pixel_values/image_hidden_states).

```python
# Common test pattern: small config, shape assertions
config = _make_config(hidden_size=256, num_hidden_layers=2)
model = SomeModule(config)
model.eval()
with torch.no_grad():
    out = model(x)
assert out.shape == (B, T, D)
```

## SigLino-specific Notes

- Uses **golden RoPE** (per-head frequency bands) for 2D position encoding, plus standard RoPE for 1D.
- Supports **MoE** (mixture of experts) layers and dense FFN layers (configurable via `first_n_layers_dense`).
- Teacher adapters (`dinov3`, `siglip2`) are trained with distillation loss but the fused `siglino` features are the main output used by DopamineVLA.
- `SigLinoImageProcessor` provides `smart_resize` to fit images to patch-aligned dimensions without padding.
- PCA visualization available via `pca_maps.py` to inspect feature quality.
