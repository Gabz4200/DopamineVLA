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

# HuggingFace Transformers integration for SigLino
# Provides PreTrainedModel/PreTrainedConfig wrappers for quantization, device_map, Hub save/load

from typing import Any, Literal

import torch
from transformers import PretrainedConfig, PreTrainedModel

from .configs import MoEArgs, SigLinoArgs
from .model import SigLino

# Mapping from HF hub config field names to our config field names
_HUB_FIELD_MAP = {
    "dim": "hidden_size",
    "n_layers": "num_hidden_layers",
    "n_heads": "num_attention_heads",
    "n_kv_heads": "num_key_value_heads",
}


class SigLinoConfig(PretrainedConfig):
    model_type = "siglino"

    ## Defaults match Siglino-70M defaults
    def __init__(
        self,
        hidden_size: int = 512,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 8,
        head_dim: int | None = 64,
        num_key_value_heads: int | None = 8,
        moe_dim: int = 0,
        moe_num_experts: int = 1,
        moe_num_shared_experts: int = 0,
        moe_top_k: int = 1,
        moe_score_before_experts: bool = False,
        moe_route_norm: bool = False,
        moe_route_scale: float = 1.0,
        moe_score_func: Literal["softmax", "sigmoid"] = "sigmoid",
        moe_activation: Literal["silu", "relu2"] = "silu",
        first_n_layers_dense: int = 12,
        ffn_dim: int | None = 2048,
        activation: str = "silu",
        channel_size: int = 3,
        spatial_patch_size: int = 16,
        temporal_patch_size: int = 1,
        enable_3d_rope: bool = True,
        rope_theta: float = 100000.0,
        rope_min_freqs: float = 1.0,
        rope_max_freqs: float = 20.0,
        max_seq_len: int = 8192,
        norm_eps: float = 1e-5,
        use_qk_norm: bool = True,
        use_tok_norm: bool = True,
        parameterized_norm: bool = False,
        n_storage_tokens: int = 4,
        teachers: tuple[str, ...] = ("siglip2", "dinov3"),
        teachers_dim: tuple[int, ...] = (1152, 1024),
        depth_init: bool = True,
        use_flex_attn: bool = True,
        **kwargs: Any,
    ) -> None:
        hidden_size = kwargs.pop("dim", hidden_size)
        num_hidden_layers = kwargs.pop("n_layers", num_hidden_layers)
        num_attention_heads = kwargs.pop("n_heads", num_attention_heads)
        num_key_value_heads = kwargs.pop("n_kv_heads", num_key_value_heads)
        ffn_dim = kwargs.pop("ffn_dim", ffn_dim)
        activation = kwargs.pop("activation", activation)
        channel_size = kwargs.pop("channel_size", channel_size)
        spatial_patch_size = kwargs.pop("spatial_patch_size", spatial_patch_size)
        temporal_patch_size = kwargs.pop("temporal_patch_size", temporal_patch_size)
        enable_3d_rope = kwargs.pop("enable_3d_rope", enable_3d_rope)
        rope_theta = kwargs.pop("rope_theta", rope_theta)
        rope_min_freqs = kwargs.pop("rope_min_freqs", rope_min_freqs)
        rope_max_freqs = kwargs.pop("rope_max_freqs", rope_max_freqs)
        max_seq_len = kwargs.pop("max_seq_len", max_seq_len)
        norm_eps = kwargs.pop("norm_eps", norm_eps)
        use_qk_norm = kwargs.pop("use_qk_norm", use_qk_norm)
        use_tok_norm = kwargs.pop("use_tok_norm", use_tok_norm)
        parameterized_norm = kwargs.pop("parameterized_norm", parameterized_norm)
        n_storage_tokens = kwargs.pop("n_storage_tokens", n_storage_tokens)
        depth_init = kwargs.pop("depth_init", depth_init)
        use_flex_attn = kwargs.pop("use_flex_attn", use_flex_attn)
        first_n_layers_dense = kwargs.pop("first_n_layers_dense", first_n_layers_dense)
        moe_dim = kwargs.pop("moe_dim", moe_dim)
        head_dim = kwargs.pop("head_dim", head_dim)
        if "teachers" in kwargs:
            teachers = kwargs.pop("teachers")
        if "teachers_dim" in kwargs:
            teachers_dim = kwargs.pop("teachers_dim")
        if "moe_args" in kwargs:
            moe_dict = kwargs.pop("moe_args")
            if isinstance(moe_dict, dict):
                moe_num_experts = moe_dict.get("num_experts", moe_num_experts)
                moe_num_shared_experts = moe_dict.get("num_shared_experts", moe_num_shared_experts)
                moe_top_k = moe_dict.get("top_k", moe_top_k)
                moe_score_before_experts = moe_dict.get(
                    "score_before_experts", moe_score_before_experts
                )
                moe_route_norm = moe_dict.get("route_norm", moe_route_norm)
                moe_route_scale = moe_dict.get("route_scale", moe_route_scale)
                moe_score_func = moe_dict.get("score_func", moe_score_func)
                moe_activation = moe_dict.get("activation", moe_activation)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.moe_dim = moe_dim
        self.moe_num_experts = moe_num_experts
        self.moe_num_shared_experts = moe_num_shared_experts
        self.moe_top_k = moe_top_k
        self.moe_score_before_experts = moe_score_before_experts
        self.moe_route_norm = moe_route_norm
        self.moe_route_scale = moe_route_scale
        self.moe_score_func = moe_score_func
        self.moe_activation = moe_activation
        self.first_n_layers_dense = first_n_layers_dense
        self.ffn_dim = ffn_dim
        self.activation = activation
        self.channel_size = channel_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.enable_3d_rope = enable_3d_rope
        self.rope_theta = rope_theta
        self.rope_min_freqs = rope_min_freqs
        self.rope_max_freqs = rope_max_freqs
        self.max_seq_len = max_seq_len
        self.norm_eps = norm_eps
        self.use_qk_norm = use_qk_norm
        self.use_tok_norm = use_tok_norm
        self.parameterized_norm = parameterized_norm
        self.n_storage_tokens = n_storage_tokens
        self.teachers = list(teachers)
        self.teachers_dim = list(teachers_dim)
        self.depth_init = depth_init
        self.use_flex_attn = use_flex_attn
        super().__init__(**kwargs)

    def to_siglino_args(self) -> SigLinoArgs:
        """Convert this HF config to a SigLinoArgs dataclass."""
        return SigLinoArgs(
            dim=self.hidden_size,
            n_layers=self.num_hidden_layers,
            n_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            n_kv_heads=self.num_key_value_heads,
            moe_dim=self.moe_dim,
            moe_args=MoEArgs(
                num_experts=self.moe_num_experts,
                num_shared_experts=self.moe_num_shared_experts,
                top_k=self.moe_top_k,
                score_func=self.moe_score_func,
                score_before_experts=self.moe_score_before_experts,
                route_norm=self.moe_route_norm,
                route_scale=self.moe_route_scale,
                activation=self.moe_activation,
            ),
            first_n_layers_dense=self.first_n_layers_dense,
            ffn_dim=self.ffn_dim,
            activation=self.activation,
            channel_size=self.channel_size,
            spatial_patch_size=self.spatial_patch_size,
            temporal_patch_size=self.temporal_patch_size,
            enable_3d_rope=self.enable_3d_rope,
            rope_theta=self.rope_theta,
            rope_min_freqs=self.rope_min_freqs,
            rope_max_freqs=self.rope_max_freqs,
            max_seq_len=self.max_seq_len,
            norm_eps=self.norm_eps,
            use_qk_norm=self.use_qk_norm,
            use_tok_norm=self.use_tok_norm,
            parameterized_norm=self.parameterized_norm,
            n_storage_tokens=self.n_storage_tokens,
            depth_init=self.depth_init,
            teachers=tuple(self.teachers),
            teachers_dim=tuple(self.teachers_dim),
            use_flex_attn=self.use_flex_attn,
        )

    @classmethod
    def from_hub_config(cls, config_dict: dict[str, Any]) -> "SigLinoConfig":
        """Create a SigLinoConfig from a hub-style config dict (dim, n_layers, etc.)."""
        mapped = {}
        for hub_name, our_name in _HUB_FIELD_MAP.items():
            if hub_name in config_dict:
                mapped[our_name] = config_dict[hub_name]
        for k, v in config_dict.items():
            if k not in _HUB_FIELD_MAP and k not in _HUB_FIELD_MAP.values() and k != "model_type":
                mapped[k] = v
        return cls(**mapped)

    @classmethod
    def from_siglino_args(cls, args: SigLinoArgs, **kwargs: Any) -> "SigLinoConfig":
        """Create a HF config from a SigLinoArgs dataclass."""
        return cls(
            hidden_size=args.dim,
            num_hidden_layers=args.n_layers,
            num_attention_heads=args.n_heads,
            head_dim=args.head_dim,
            num_key_value_heads=args.n_kv_heads,
            moe_dim=args.moe_dim,
            moe_num_experts=args.moe_args.num_experts,
            moe_num_shared_experts=args.moe_args.num_shared_experts,
            moe_top_k=args.moe_args.top_k,
            moe_score_before_experts=args.moe_args.score_before_experts,
            moe_route_norm=args.moe_args.route_norm,
            moe_route_scale=args.moe_args.route_scale,
            moe_score_func=args.moe_args.score_func,
            moe_activation=args.moe_args.activation,
            first_n_layers_dense=args.first_n_layers_dense,
            ffn_dim=args.ffn_dim,
            activation=args.activation,
            channel_size=args.channel_size,
            spatial_patch_size=args.spatial_patch_size,
            temporal_patch_size=args.temporal_patch_size,
            enable_3d_rope=args.enable_3d_rope,
            rope_theta=args.rope_theta,
            rope_min_freqs=args.rope_min_freqs,
            rope_max_freqs=args.rope_max_freqs,
            max_seq_len=args.max_seq_len,
            norm_eps=args.norm_eps,
            use_qk_norm=args.use_qk_norm,
            use_tok_norm=args.use_tok_norm,
            parameterized_norm=args.parameterized_norm,
            n_storage_tokens=args.n_storage_tokens,
            depth_init=args.depth_init,
            teachers=args.teachers,
            teachers_dim=args.teachers_dim,
            use_flex_attn=args.use_flex_attn,
            **kwargs,
        )


class SigLinoPreTrainedModel(PreTrainedModel):
    """Base class for SigLino models with HF integration."""

    config_class = SigLinoConfig
    base_model_prefix = ""
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock", "Attention", "MoE", "FeedForward"]
    _supports_sdpa = False
    _supports_flash_attn = False
    _supports_flex_attn = False
    _supports_attention_backend = False
    _keys_to_ignore_on_load_missing = [
        "model.freqs_cis",  # non-persistent buffer, recomputed via _post_init
        "model.freqs_cis_golden",  # precomputed in _post_init
    ]

    def _init_weights(self, module: torch.nn.Module) -> None:
        pass  # Weight init is handled by SigLino.init_weights()


class SigLinoHFModel(SigLinoPreTrainedModel):
    """HF-compatible SigLino vision model.

    Wraps the core SigLino model with PreTrainedModel for quantization,
    device_map, and Hub save/load support.
    """

    def __init__(self, config: SigLinoConfig) -> None:
        super().__init__(config)
        siglino_args = config.to_siglino_args()
        self.model: SigLino = SigLino(siglino_args)
        self.model.init_weights()  # Initialize empty params, calls _post_init for RoPE buffers
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        **kwargs: object,
    ) -> dict[str, dict[str, torch.Tensor]]:
        compile_ = kwargs.pop("compile", None)
        return self.model(
            pixel_values=pixel_values,
            padding_mask=padding_mask,
            spatial_shapes=spatial_shapes,
            compile=compile_,
        )

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.img_projector

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.model.img_projector = value  # pyrefly: ignore[bad-assignment]
