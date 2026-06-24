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

import torch

from .configs import siglino_configs
from .image_processor import SigLinoImageProcessor
from .model import SigLino


def load_siglino_model(
    checkpoint_path: str | None,
    config_name: str = "siglino-0.3B",
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    **kwargs,
) -> tuple[SigLino, SigLinoImageProcessor]:
    """
    Load a SigLino model from a checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint
        config_name: Name of the model configuration
        device: Device to load the model on (default: cuda if available, else cpu)
        dtype: Optional dtype to cast model weights to (e.g. torch.bfloat16)

    Returns:
        Tuple of (model, image_processor)
    """
    # Auto-detect device: CUDA if available, else CPU
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get configuration
    if config_name in siglino_configs:
        args = siglino_configs[config_name]
    else:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(siglino_configs.keys())}")

    # Create model
    model = SigLino(args)
    # Initialize weights (must be called explicitly - SigLino.__init__ uses torch.empty)
    model.init_weights()

    # Load checkpoint weights (None = random init)
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)

    if dtype is None:
        model = model.to(device=device)
    else:
        model = model.to(device=device, dtype=dtype)
    model.eval()

    # Create image processor
    image_processor = SigLinoImageProcessor(patch_size=args.spatial_patch_size, **kwargs)

    return model, image_processor


def quantize_cpu_model(model: torch.nn.Module) -> torch.nn.Module:
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
