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

"""Configurations for DopamineVLA — vision wrapper and top-level model config."""

from typing import Any

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig
from transformers.utils import logging

from dopaminevla.models.siglino.siglino import SigLinoConfig

logger = logging.get_logger(__name__)


class DopamineVLAVisionConfig(SigLinoConfig):
    """Vision configuration for DopamineVLA — a thin wrapper around SigLinoConfig."""

    model_type = "dopaminevla_vision"

    def __init__(
        self,
        vision_feature_layers: int = 1,
        **kwargs: Any,
    ) -> None:
        self.vision_feature_layers = vision_feature_layers
        super().__init__(**kwargs)


class DopamineVLAConfig(PretrainedConfig):
    """Configuration for DopamineVLA vision-language-action model.

    Composes a vision config (SigLino) and a text config (any HF LM, default
    Llama) with a Perceiver-based connector between them.
    """

    model_type = "dopaminevla"
    sub_configs = {"text_config": AutoConfig, "vision_config": SigLinoConfig}  # pyrefly: ignore[bad-assignment]

    def __init__(
        self,
        vision_config: dict | PretrainedConfig | None = None,
        text_config: dict | PretrainedConfig | None = None,
        use_cache: bool = True,
        image_token_id: int = 128_257,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = 128_002,
        scale_factor: int = 2,
        attn_implementation: str = "eager",
        vision_connector_n_latents: int = 128,
        vision_connector_n_layers: int = 3,
        vision_connector_n_heads: int = 16,
        vision_connector_head_dim: int = 96,
        vision_connector_n_kv_heads: int | None = None,
        vision_connector_ffn_mult: int = 4,
        vision_connector_attn_dropout: float = 0.0,
        vision_connector_rms_eps: float = 1e-6,
        vision_feature_layers: int = 1,
        # Action head params
        num_action_queries: int = 16,
        action_embed_dim: int = 512,
        action_swa_layers: int = 4,
        action_swa_heads: int = 8,
        action_swa_window_size: int = 64,
        action_swa_ffn_mult: int = 4,
        action_delta_dim: int = 24,
        action_cross_attention_heads: int = 8,
        action_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        # Resolve vision config — forward vision_feature_layers
        if vision_config is None:
            vision_config = DopamineVLAVisionConfig(vision_feature_layers=vision_feature_layers)
        elif isinstance(vision_config, dict):
            vision_config.setdefault("vision_feature_layers", vision_feature_layers)
            vision_config = DopamineVLAVisionConfig(**vision_config)

        # Resolve text config
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "llama")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            logger.info("text_config is None, defaulting to Llama config")
            text_config = CONFIG_MAPPING["llama"](
                rms_norm_eps=1e-5,
                # Standard Llama 3 vocab; pad_token_id must be < vocab_size
                vocab_size=128_256,
                pad_token_id=pad_token_id,
            )

        # Store resolved configs as instance attributes
        self.vision_config: DopamineVLAVisionConfig | PretrainedConfig = vision_config  # pyrefly: ignore[bad-assignment]
        self.text_config: PretrainedConfig = text_config

        # Stacked vision layers (1 = last layer only, >1 = last N layers, -1 = all)
        self.vision_feature_layers = vision_feature_layers

        # Connector params (used by DopamineVLAConnector)
        self.vision_connector_n_latents = vision_connector_n_latents
        self.vision_connector_n_layers = vision_connector_n_layers
        self.vision_connector_n_heads = vision_connector_n_heads
        self.vision_connector_head_dim = vision_connector_head_dim
        self.vision_connector_n_kv_heads = vision_connector_n_kv_heads
        self.vision_connector_ffn_mult = vision_connector_ffn_mult
        self.vision_connector_attn_dropout = vision_connector_attn_dropout
        self.vision_connector_rms_eps = vision_connector_rms_eps

        # Action head params
        self.num_action_queries = num_action_queries
        self.action_embed_dim = action_embed_dim
        self.action_swa_layers = action_swa_layers
        self.action_swa_heads = action_swa_heads
        self.action_swa_window_size = action_swa_window_size
        self.action_swa_ffn_mult = action_swa_ffn_mult
        self.action_delta_dim = action_delta_dim
        self.action_cross_attention_heads = action_cross_attention_heads
        self.action_token_id = action_token_id

        # Store generation-relevant config BEFORE super().__init__ because
        # PretrainedConfig pops all GenerationConfig params from kwargs (transformers 5.x).
        self.use_cache = use_cache

        super().__init__(
            image_token_id=image_token_id,
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            scale_factor=scale_factor,
            attn_implementation=attn_implementation,
            **kwargs,
        )


__all__ = [
    "DopamineVLAVisionConfig",
    "DopamineVLAConfig",
]
