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

"""Output dataclasses for DopamineVLA."""

from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput


@dataclass
class DopamineVLABaseModelOutputWithPast(ModelOutput):
    """Base model output with past for DopamineVLA.

    Fields match standard HF VLM output conventions but are defined
    here so we own them.
    """

    last_hidden_state: torch.Tensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    image_hidden_states: torch.Tensor | None = None


@dataclass
class DopamineVLACausalLMOutputWithPast(ModelOutput):
    """Causal LM output with past for DopamineVLA (action-head logits added later)."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    image_hidden_states: torch.Tensor | None = None
    action: torch.Tensor | None = None
    delta_actions: torch.Tensor | None = None


@dataclass
class DopamineVLAActionOutput(ModelOutput):
    """Action head specific output."""

    action: torch.Tensor | None = None
    delta_actions: torch.Tensor | None = None


__all__ = [
    "DopamineVLABaseModelOutputWithPast",
    "DopamineVLACausalLMOutputWithPast",
    "DopamineVLAActionOutput",
]
