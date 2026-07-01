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

"""DopamineVLA — self-contained vision-language-action model.

Depends only on PyTorch and base transformers (PreTrainedConfig, PreTrainedModel,
ModelOutput, GenerationMixin).  Vision encoder = SigLino (our own).  No SmolVLM,
Idefics, or other HF model subclasses.
"""

from .action import ActionHead, ActionSWABlock, Time2Vec
from .base import DopamineVLAPreTrainedModel
from .configuration_dopaminevla import DopamineVLAConfig, DopamineVLAVisionConfig
from .connector import (
    DopamineVLAConnector,
    DopamineVLAPerceiverAttention,
    DopamineVLAPerceiverLayer,
)
from .generation import DopamineVLAForConditionalGeneration
from .merger import DopamineVLAInputsMerger
from .model import DopamineVLAModel
from .outputs import (
    DopamineVLAActionOutput,
    DopamineVLABaseModelOutputWithPast,
    DopamineVLACausalLMOutputWithPast,
)
from .vision import DopamineVLAVisionTransformer

__all__ = [
    "DopamineVLAVisionConfig",
    "DopamineVLAConfig",
    "DopamineVLAPreTrainedModel",
    "DopamineVLAVisionTransformer",
    "DopamineVLABaseModelOutputWithPast",
    "DopamineVLACausalLMOutputWithPast",
    "DopamineVLAActionOutput",
    "DopamineVLAInputsMerger",
    "DopamineVLAConnector",
    "DopamineVLAPerceiverAttention",
    "DopamineVLAPerceiverLayer",
    "DopamineVLAModel",
    "DopamineVLAForConditionalGeneration",
    "ActionHead",
    "ActionSWABlock",
    "Time2Vec",
]
