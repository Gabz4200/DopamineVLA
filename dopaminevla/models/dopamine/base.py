# Copyright (c) 2025 Gabriel Amaral
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

"""PreTrainedModel base class for DopamineVLA."""

from transformers.modeling_utils import PreTrainedModel

from .configuration_dopaminevla import DopamineVLAConfig


class DopamineVLAPreTrainedModel(PreTrainedModel):
    """Base class for DopamineVLA models, inheriting from HF PreTrainedModel."""

    config_class = DopamineVLAConfig
    base_model_prefix = "model"
    input_modalities = ["image", "text"]
    supports_gradient_checkpointing = True
    _no_split_modules: list[str] | set[str] | None = []
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _supports_cache_class = True


__all__ = ["DopamineVLAPreTrainedModel"]
