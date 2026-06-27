# Copyright (c) 2025 TII (Technology Innovation Institute)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Edited by Gabriel Amaral

# Utilities for Falcon Vision
# Model loading and image preprocessing without tokenizer dependency
# Includes config/checkpoint resolution: auto-detect config from hub checkpoint,
# auto-resolve hub checkpoint from config, and validate they match.

import os
from typing import Any, TypeVar

import torch

from .configs import SigLinoArgs, siglino_configs
from .image_processor import SigLinoImageProcessor
from .model import SigLino

# Mapping from local config names to HuggingFace Hub model IDs
_CONFIG_TO_HUB_ID: dict[str, str] = {
    "dense-30M": "tiiuae/siglino-30M",
    "dense-70M": "tiiuae/siglino-70M",
    "dense-0.6B": "tiiuae/siglino-0.6B",
    "siglino-0.15B": "tiiuae/siglino-0.15B",
    "siglino-0.3B": "tiiuae/siglino-0.3B",
}


def _is_hub_id(path: str) -> bool:
    """Check if a path is a HuggingFace Hub model ID (org/name, not a local file)."""
    # Hub IDs have no file extension and match ``org/name`` (no leading slash).
    if "." in path or "/" not in path or path.startswith("/"):
        return False
    if os.path.isdir(path) or os.path.isfile(path):
        return False
    return True


def _download_hub_checkpoint(hub_id: str) -> str:
    """Download a hub model checkpoint and return the local path."""
    from huggingface_hub import hf_hub_download  # already a transitive dep

    for filename in ("model.safetensors", "pytorch_model.bin"):
        try:
            return hf_hub_download(repo_id=hub_id, filename=filename)
        except Exception:
            continue
    raise FileNotFoundError(f"No checkpoint file (safetensors or bin) found for {hub_id}")


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Load state dict from a checkpoint file, supporting both safetensors and torch."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _read_hub_config_args(hub_id: str) -> SigLinoArgs:
    """Read a hub model's config.json and convert to SigLinoArgs."""
    from .hf_integration import SigLinoConfig

    hf_config = SigLinoConfig.from_pretrained(hub_id)
    return hf_config.to_siglino_args()


def _find_matching_config_name(args: SigLinoArgs) -> str | None:
    """Find a local config name whose SigLinoArgs matches the given args."""
    for name, candidate in siglino_configs.items():
        if (
            candidate.dim == args.dim
            and candidate.n_layers == args.n_layers
            and candidate.n_heads == args.n_heads
            and candidate.head_dim == args.head_dim
            and candidate.n_kv_heads == args.n_kv_heads
            and candidate.moe_dim == args.moe_dim
            and candidate.first_n_layers_dense == args.first_n_layers_dense
            and candidate.ffn_dim == args.ffn_dim
            and candidate.activation == args.activation
            and candidate.spatial_patch_size == args.spatial_patch_size
            and candidate.moe_args == args.moe_args
        ):
            return name
    return None


def _validate_config_checkpoint_match(
    config_name: str,
    config_args: SigLinoArgs,
    checkpoint_path: str,
    checkpoint_args: SigLinoArgs,
) -> None:
    """Raise ValueError if config and checkpoint describe different architectures."""
    mismatches: list[str] = []
    checks = {
        "dim": (config_args.dim, checkpoint_args.dim),
        "n_layers": (config_args.n_layers, checkpoint_args.n_layers),
        "n_heads": (config_args.n_heads, checkpoint_args.n_heads),
        "head_dim": (config_args.head_dim, checkpoint_args.head_dim),
        "n_kv_heads": (config_args.n_kv_heads, checkpoint_args.n_kv_heads),
        "moe_dim": (config_args.moe_dim, checkpoint_args.moe_dim),
        "ffn_dim": (config_args.ffn_dim, checkpoint_args.ffn_dim),
        "activation": (config_args.activation, checkpoint_args.activation),
    }
    for key, (c_val, chk_val) in checks.items():
        if c_val != chk_val:
            mismatches.append(f"  {key}: config={c_val}, checkpoint={chk_val}")

    if mismatches:
        raise ValueError(
            f"Config '{config_name}' does not match checkpoint '{checkpoint_path}':\n"
            + "\n".join(mismatches)
        )


def load_siglino_model(
    checkpoint_path: str | None = None,
    config_name: str | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    resolve: bool = False,
    **kwargs: Any,
) -> tuple[SigLino, SigLinoImageProcessor]:
    """Load a SigLino model from a checkpoint and/or config.

    When *resolve* is ``True``, the function handles the following scenarios:

    1. **Config only** (``config_name`` given, ``checkpoint_path=None``):
       Auto-resolves the matching HuggingFace Hub checkpoint.

    2. **Checkpoint only** (``checkpoint_path`` given, ``config_name=None``):
       Infers a matching config from the checkpoint's hub metadata. Warns if
       no exact local config matches, but still uses the checkpoint data.

    3. **Both given and match**: Normal operation.

    4. **Both given and mismatch**: Raises ``ValueError``.

    When *resolve* is ``False`` (default), the function behaves as before:
    ``checkpoint_path=None`` means random initialization,
    ``config_name`` must be in ``siglino_configs``.

    Args:
        checkpoint_path: Path to the model checkpoint, or HF Hub ID (e.g.
            ``"tiiuae/siglino-30M"``). ``None`` means random init unless
            *resolve* is ``True``, in which case a hub ID is derived from
            *config_name*.
        config_name: Name of the model configuration from ``siglino_configs``
            (e.g. ``"dense-30M"``). ``None`` means auto-detect from
            *checkpoint_path* when *resolve* is ``True``.
        device: Device to load the model on (default: cuda if available, else cpu).
        dtype: Optional dtype to cast model weights to (e.g. torch.bfloat16).
        resolve: Enable automatic config/checkpoint resolution and validation.

    Returns:
        Tuple of (model, image_processor).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Resolution phase (resolve=True) ──────────────────────────────────
    effective_ckpt: str | None = checkpoint_path
    effective_config: str | None = config_name
    inferred_args: SigLinoArgs | None = None
    if resolve:
        # (1) Config only → resolve hub checkpoint
        if effective_ckpt is None and effective_config is not None:
            if effective_config in _CONFIG_TO_HUB_ID:
                effective_ckpt = _CONFIG_TO_HUB_ID[effective_config]
                print(f"Auto-resolved checkpoint: {effective_ckpt}")

        # (2) Checkpoint only → infer config from hub metadata
        if effective_ckpt is not None and effective_config is None:
            if _is_hub_id(effective_ckpt):
                hub_args = _read_hub_config_args(effective_ckpt)
                match = _find_matching_config_name(hub_args)
                if match is not None:
                    print(f"Auto-detected config: {match}")
                    effective_config = match
                else:
                    print(
                        (
                        f"Warning: No matching local config for "
                        f"'{effective_ckpt}', using checkpoint metadata directly"
                    )
                    )
                    inferred_args = hub_args
            elif os.path.isfile(effective_ckpt):
                raise ValueError(
                    "Cannot infer config from a local checkpoint file. "
                    "Please provide --config_name explicitly."
                )

        # (3) Both given → validate match
        if config_name is not None and checkpoint_path is not None:
            if _is_hub_id(checkpoint_path):
                chk_args = _read_hub_config_args(checkpoint_path)
                _validate_config_checkpoint_match(
                    config_name,
                    siglino_configs[config_name],
                    checkpoint_path,
                    chk_args,
                )

    # Override with resolved values
    checkpoint_path = effective_ckpt
    config_name = effective_config

    # ── Loading phase ────────────────────────────────────────────────────

    # Get SigLinoArgs
    args: SigLinoArgs
    if config_name is not None:
        if config_name in siglino_configs:
            args = siglino_configs[config_name]
        else:
            raise ValueError(
                f"Unknown config: {config_name}. Available: {list(siglino_configs.keys())}"
            )
    elif inferred_args is not None:
        # No local config matched; use args read from hub metadata directly
        args = inferred_args
        config_name = f"(inferred from {checkpoint_path})"
    else:
        raise ValueError("Either config_name or checkpoint_path must be provided")

    # Create model
    model = SigLino(args)
    model.init_weights()

    # Download hub checkpoint if needed
    if checkpoint_path is not None and _is_hub_id(checkpoint_path):
        checkpoint_path = _download_hub_checkpoint(checkpoint_path)

    # Load checkpoint weights (None = random init)
    if checkpoint_path is not None:
        state_dict = _load_state_dict(checkpoint_path)
        model.load_state_dict(state_dict)

    if dtype is None:
        model = model.to(device=device)
    else:
        model = model.to(device=device, dtype=dtype)
    model.eval()

    # Create image processor
    image_processor = SigLinoImageProcessor(patch_size=args.spatial_patch_size, **kwargs)

    return model, image_processor


_M = TypeVar("_M", bound=torch.nn.Module)


def quantize_cpu_model(model: _M) -> _M:
    """Apply torchao INT8 dynamic quantization for CPU inference."""
    try:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

        quantize_(model, Int8DynamicActivationInt8WeightConfig())
        print("Applied torchao INT8 dynamic quantization for CPU")
    except ImportError:
        print("torchao not available, skipping CPU quantization")
    except Exception as e:
        print(f"CPU quantization failed ({e}), running unquantized")
    return model
