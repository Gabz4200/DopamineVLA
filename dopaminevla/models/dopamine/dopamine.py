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

from dataclasses import dataclass
from typing import Any, Tuple, Unpack

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import CONFIG_MAPPING, AutoConfig, AutoModel
from transformers.utils import (
    TransformersKwargs,
    can_return_tuple,
    logging,
    torch_compilable_check,
)

from dopaminevla.models.siglino.siglino import SigLinoConfig
from dopaminevla.models.siglino.siglino.hf_integration import SigLinoHFModel

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


class DopamineVLAVisionConfig(SigLinoConfig):
    """Vision configuration for DopamineVLA — a thin wrapper around SigLinoConfig."""

    model_type = "dopaminevla_vision"


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
        vision_connector_n_latents: int = 64,
        vision_connector_n_layers: int = 3,
        vision_connector_n_heads: int = 16,
        vision_connector_head_dim: int = 96,
        vision_connector_n_kv_heads: int | None = None,
        vision_connector_ffn_mult: int = 4,
        vision_connector_attn_dropout: float = 0.0,
        vision_connector_rms_eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        # Resolve vision config
        if vision_config is None:
            vision_config = DopamineVLAVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = DopamineVLAVisionConfig(**vision_config)

        # Resolve text config
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "llama")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            logger.info("text_config is None, defaulting to Llama config")
            text_config = CONFIG_MAPPING["llama"](
                rms_norm_eps=1e-5,
                pad_token_id=pad_token_id,
            )

        # Store resolved configs as instance attributes
        self.vision_config = vision_config
        self.text_config = text_config

        # Connector params (used by DopamineVLAConnector)
        self.vision_connector_n_latents = vision_connector_n_latents
        self.vision_connector_n_layers = vision_connector_n_layers
        self.vision_connector_n_heads = vision_connector_n_heads
        self.vision_connector_head_dim = vision_connector_head_dim
        self.vision_connector_n_kv_heads = vision_connector_n_kv_heads
        self.vision_connector_ffn_mult = vision_connector_ffn_mult
        self.vision_connector_attn_dropout = vision_connector_attn_dropout
        self.vision_connector_rms_eps = vision_connector_rms_eps

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


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DopamineVLABaseModelOutputWithPast(ModelOutput):
    """Base model output with past for DopamineVLA.

    Fields match the upstream conventions (SmolVLM / Idefics) but are defined
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


# ---------------------------------------------------------------------------
# PreTrainedModel base
# ---------------------------------------------------------------------------


class DopamineVLAPreTrainedModel(PreTrainedModel):
    """Base class for DopamineVLA models, inheriting from HF PreTrainedModel."""

    config_class = DopamineVLAConfig
    base_model_prefix = "model"
    input_modalities = ["image", "text"]
    supports_gradient_checkpointing = True
    _no_split_modules = ["DopamineVLAVisionAttention", "DopamineVLAEncoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _supports_cache_class = True


# ---------------------------------------------------------------------------
# Vision Transformer — wrapper around SigLinoHFModel
# ---------------------------------------------------------------------------


class DopamineVLAVisionTransformer(DopamineVLAPreTrainedModel):
    """Multi-view vision encoder wrapping SigLinoHFModel.

    Takes an image and produces three overlapping views (full, left crop, right
    crop), forward each through SigLino, and returns a tuple of patch-feature
    tensors — one per view — for the Perceiver connector to fuse.
    """

    def __init__(self, config: DopamineVLAVisionConfig) -> None:
        super().__init__(config)
        self.vision_model = SigLinoHFModel(config)
        self.patch_size = config.spatial_patch_size
        self.post_init()

    def pre_process_views(
        self,
        pixel_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        overlap_pixels: int = 32,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        b, _, h, w = pixel_values.shape

        if padding_mask is None:
            padding_mask = torch.ones(
                (b, h // self.patch_size, w // self.patch_size),
                dtype=torch.bool,
                device=pixel_values.device,
            )

        w_patches = padding_mask.size(-1)
        patch_mid = w_patches // 2
        patch_overlap = min(overlap_pixels // self.patch_size, patch_mid)

        left_patch_end = patch_mid + patch_overlap
        right_patch_start = patch_mid - patch_overlap

        left_pixel_end = left_patch_end * self.patch_size
        right_pixel_start = right_patch_start * self.patch_size

        return (
            (
                pixel_values,
                pixel_values[..., :left_pixel_end],
                pixel_values[..., right_pixel_start:],
            ),
            (
                padding_mask,
                padding_mask[..., :left_patch_end],
                padding_mask[..., right_patch_start:],
            ),
        )

    def _forward_branch(
        self,
        pixel_values: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = pixel_values.shape

        out = self.vision_model(
            pixel_values=pixel_values,
            padding_mask=padding_mask.reshape(b, -1),
            spatial_shapes=torch.tensor(
                [[h // self.patch_size, w // self.patch_size]] * b,
                dtype=torch.long,
                device=pixel_values.device,
            ),
        )
        return out["patch_features"]["siglino"]

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Returns a tuple of patch-feature tensors, one per view."""
        pixel_views, mask_views = self.pre_process_views(
            pixel_values=pixel_values,
            padding_mask=patch_attention_mask,
        )

        return tuple(
            self._forward_branch(p, m) for p, m in zip(pixel_views, mask_views, strict=True)
        )


# ---------------------------------------------------------------------------
# Connector — Perceiver-based multi-view resampler
# ---------------------------------------------------------------------------


class DopamineVLARMSNorm(nn.Module):
    """RMSNorm — identical to Idefics2RMSNorm, isolated so we own it."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


class DopamineVLAPerceiverAttention(nn.Module):
    """
    Cross-attention block: learnable latents (queries) attend to the
    concatenation of [context, latents] (keys/values).
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int | None = None,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.scale = head_dim**-0.5
        self.attn_dropout = attn_dropout

        self.q_proj = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_size, bias=False)

    @staticmethod
    def _repeat_kv(t: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return t
        B, H, L, D = t.shape
        return t[:, :, None, :, :].expand(B, H, n_rep, L, D).reshape(B, H * n_rep, L, D)

    def forward(
        self,
        latents: torch.Tensor,  # (B, n_latents, hidden_size)
        context: torch.Tensor,  # (B, seq,       hidden_size)
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, q_len, _ = latents.shape
        kv_len = q_len + context.size(1)

        kv_input = torch.cat([context, latents], dim=1)  # (B, seq+n_latents, D)

        q = self.q_proj(latents)
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)

        q = q.view(B, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, kv_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, kv_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        k = self._repeat_kv(k, self.n_kv_groups)
        v = self._repeat_kv(v, self.n_kv_groups)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            scale=self.scale,
        )

        attn_out = attn_out.transpose(1, 2).reshape(B, q_len, self.n_heads * self.head_dim)
        return self.o_proj(attn_out)


class DopamineVLAPerceiverLayer(nn.Module):
    """One Perceiver block: pre-norm cross-attn + pre-norm FFN + residuals."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int | None = None,
        ffn_mult: int = 4,
        attn_dropout: float = 0.0,
        rms_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.latents_norm = DopamineVLARMSNorm(hidden_size, eps=rms_eps)
        self.context_norm = DopamineVLARMSNorm(hidden_size, eps=rms_eps)
        self.cross_attn = DopamineVLAPerceiverAttention(
            hidden_size, n_heads, head_dim, n_kv_heads, attn_dropout
        )
        self.post_attn_norm = DopamineVLARMSNorm(hidden_size, eps=rms_eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ffn_mult, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size * ffn_mult, hidden_size, bias=False),
        )

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latents = latents + self.cross_attn(
            self.latents_norm(latents),
            self.context_norm(context),
            attention_mask=attention_mask,
        )
        latents = latents + self.ffn(self.post_attn_norm(latents))
        return latents


class DopamineVLAConnector(nn.Module):
    """Multi-view Perceiver connector.

    Fuses an arbitrary number of visual-encoder views (original, left crop,
    right crop, …) into exactly ``n_latents`` fixed-length tokens via
    Perceiver-style cross-attention.

    Parameters are read from the ``vision_connector_*`` fields of
    ``DopamineVLAConfig``.
    """

    def __init__(self, config: DopamineVLAConfig) -> None:
        super().__init__()

        self.n_latents = config.vision_connector_n_latents
        vision_hidden_size = config.vision_config.hidden_size
        text_hidden_size = config.text_config.hidden_size

        n_layers = config.vision_connector_n_layers
        n_heads = config.vision_connector_n_heads
        head_dim = config.vision_connector_head_dim
        n_kv_heads = config.vision_connector_n_kv_heads
        ffn_mult = config.vision_connector_ffn_mult
        attn_dropout = config.vision_connector_attn_dropout
        rms_eps = config.vision_connector_rms_eps

        # Learnable query embeddings
        self.latents = nn.Parameter(
            torch.empty(self.n_latents, vision_hidden_size).normal_(std=0.02)
        )

        self.layers = nn.ModuleList(
            [
                DopamineVLAPerceiverLayer(
                    hidden_size=vision_hidden_size,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    n_kv_heads=n_kv_heads,
                    ffn_mult=ffn_mult,
                    attn_dropout=attn_dropout,
                    rms_eps=rms_eps,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = DopamineVLARMSNorm(vision_hidden_size, eps=rms_eps)
        self.modality_projection = nn.Linear(vision_hidden_size, text_hidden_size, bias=False)

    def forward(
        self,
        view_hidden_states: Tuple[torch.Tensor, ...],
        attention_masks: Tuple[torch.Tensor | None, ...] | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        view_hidden_states : tuple of Tensor, each (B, L_i, vision_hidden_size)
            Vision-encoder outputs for each view.  L_i may differ across views.
        attention_masks : tuple of optional bool Tensor, each (B, L_i), optional
            Padding masks for each view (True = valid).  ``None`` = all valid.

        Returns
        -------
        Tensor shape ``(B, n_latents, text_hidden_size)``
        """
        B = view_hidden_states[0].size(0)

        # 1. Concatenate all views into one context sequence.
        if len(view_hidden_states) == 1:
            context = view_hidden_states[0]
        else:
            context = torch.cat(view_hidden_states, dim=1)  # (B, sum(L_i), D_vis)

        # 2. Build a unified attention mask for the full context + latents.
        attn_mask = None
        if attention_masks is not None:
            valid_parts: list[torch.Tensor] = []
            for i, m in enumerate(attention_masks):
                if m is None:
                    L_i = view_hidden_states[i].size(1)
                    valid_parts.append(torch.ones(B, L_i, dtype=torch.bool, device=context.device))
                else:
                    valid_parts.append(m)

            ctx_mask = torch.cat(valid_parts, dim=1)
            lat_mask = torch.ones(B, self.n_latents, dtype=torch.bool, device=context.device)
            full_mask = torch.cat([ctx_mask, lat_mask], dim=1)  # (B, kv_len)

            full_4d = full_mask[:, None, None, :]  # (B, 1, 1, kv_len) broadcast over q_len
            attn_mask = torch.full(
                (B, 1, self.n_latents, full_mask.size(1)),
                float("-inf"),
                dtype=context.dtype,
                device=context.device,
            )
            attn_mask = attn_mask.masked_fill(full_4d, 0.0)

        # 3. Expand latents over batch (zero-copy view).
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, n_latents, D_vis)

        for layer in self.layers:
            latents = layer(latents, context, attention_mask=attn_mask)

        latents = self.norm(latents)
        latents = self.modality_projection(latents)  # (B, n_latents, D_text)
        return latents


# ---------------------------------------------------------------------------
# Core Model
# ---------------------------------------------------------------------------


class DopamineVLAModel(DopamineVLAPreTrainedModel):
    """DopamineVLA model: vision encoder + connector + text decoder."""

    def __init__(self, config: DopamineVLAConfig) -> None:
        super().__init__(config)
        self.padding_idx = self.config.text_config.pad_token_id
        self.vocab_size = self.config.text_config.vocab_size

        self.vision_model: DopamineVLAVisionTransformer = DopamineVLAVisionTransformer._from_config(
            config.vision_config
        )
        self.connector = DopamineVLAConnector(config)
        self.text_model = AutoModel.from_config(config.text_config)

        self.image_token_id = self.config.image_token_id
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.text_model.set_input_embeddings(value)

    def inputs_merger(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        image_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Merge image hidden states into the token-embedding sequence.

        ``image_hidden_states`` has shape ``(N_images, n_latents, text_hidden_size)``,
        where ``n_latents`` is the Perceiver's fixed token budget per image.
        """
        if input_ids is None:
            raise ValueError("input_ids is required for image token merging")

        _, patch_size, _ = image_hidden_states.shape

        image_mask = input_ids == self.config.image_token_id

        num_image_tokens = image_mask.sum(dim=1)
        torch_compilable_check(
            torch.all(num_image_tokens % patch_size == 0),
            "At least one sample has <image> tokens not divisible by patch_size.",
        )
        blocks_per_sample = num_image_tokens // patch_size

        offsets = torch.nn.functional.pad(blocks_per_sample.cumsum(dim=0), (1, 0), value=0)
        block_offset = offsets[:-1]
        row_cum = image_mask.cumsum(dim=-1)
        chunk_idx = (row_cum - 1) // patch_size
        local_idx = (row_cum - 1) % patch_size
        block_idx = block_offset.unsqueeze(1) + chunk_idx

        image_embeds = torch.zeros_like(inputs_embeds)
        image_embeds[image_mask] = image_hidden_states[
            block_idx[image_mask], local_idx[image_mask], :
        ]

        merged_embeds = torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds)
        return merged_embeds

    @can_return_tuple
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """Encode images into continuous embeddings ready for the LLM.

        Returns connector output ``(total_real_images, n_latents, text_hidden_size)``.
        The caller merges these into the token embedding stream via ``inputs_merger``.
        """
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.to(dtype=self.dtype)
        pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

        # Remove padding images (all-zero tensors).
        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image

        real_images_inds[0] |= ~torch.any(real_images_inds)

        pixel_values = pixel_values[real_images_inds].contiguous()

        if pixel_attention_mask is None:
            attn_mask = torch.ones(
                size=[pixel_values.shape[i] for i in (0, 2, 3)],
                dtype=torch.bool,
                device=pixel_values.device,
            )
        else:
            attn_mask = pixel_attention_mask.view(
                batch_size * num_images, *pixel_attention_mask.shape[2:]
            )
            attn_mask = attn_mask[real_images_inds].contiguous()

        patch_size = self.config.vision_config.spatial_patch_size
        patches_subgrid = attn_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

        # Vision model returns tuple of tensors (one per view).
        view_hidden_states = self.vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        )

        # Connector fuses views into fixed-length token set.
        image_features = self.connector(view_hidden_states)
        return image_features

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        image_hidden_states: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, ...] | DopamineVLABaseModelOutputWithPast:
        r"""
        pixel_attention_mask (:class:`torch.Tensor`, shape
            ``(batch_size, image_size, image_size)``, *optional*):
            Mask to avoid attending to padding pixel indices.
        image_hidden_states (:class:`torch.Tensor`, shape
            ``(batch_size, num_channels, image_size, image_size)``, *optional*):
            Pre-computed hidden states of the image encoder after modality
            projection (passed in for generation, avoiding re-encoding).
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. "
                "Setting `use_cache=False`..."
            )
            use_cache = False

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(input_ids.device)

        # ---- Visual inputs integration ----
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError(
                "You cannot specify both pixel_values and image_hidden_states at the same time"
            )

        if pixel_values is not None:
            image_hidden_states = self.get_image_features(
                pixel_values, pixel_attention_mask, return_dict=True
            )
            image_hidden_states = image_hidden_states.to(inputs_embeds.device)
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(
                dtype=self.dtype, device=inputs_embeds.device
            )

        if image_hidden_states is not None:
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return DopamineVLABaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


# ---------------------------------------------------------------------------
# Conditional Generation (language-modeling head)
# ---------------------------------------------------------------------------


class DopamineVLAForConditionalGeneration(DopamineVLAPreTrainedModel, GenerationMixin):
    """DopamineVLA with a language modeling head (and, in the future, action head)."""

    _tied_weights_keys = {"lm_head.weight": "model.text_model.embed_tokens.weight"}

    def __init__(self, config: DopamineVLAConfig) -> None:
        super().__init__(config)
        self.model = DopamineVLAModel(config)
        self.image_token_id = self.config.image_token_id
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.vocab_size = config.text_config.vocab_size
        self.model.text_model.generation_config = GenerationConfig.from_model_config(config)

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.text_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.text_model.set_input_embeddings(value)

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """Encode images into continuous embeddings.  Delegates to ``self.model``."""
        return self.model.get_image_features(
            pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, **kwargs
        )

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        image_hidden_states: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.Tensor | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...] | DopamineVLACausalLMOutputWithPast:
        r"""
        labels (:class:`torch.Tensor`, shape ``(batch_size, sequence_length)``, *optional*):
            Labels for masked language modeling.  Values in ``[0, vocab_size)``
            are prediction targets; values equal to ``image_token_id`` are ignored.
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs[0]
        if logits_to_keep is None:
            slice_indices = slice(None)
        elif isinstance(logits_to_keep, int):
            slice_indices = slice(-logits_to_keep, None)
        else:
            slice_indices = logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

        return DopamineVLACausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    # pyrefly: ignore[bad-override]
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        image_hidden_states: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # KV-cache: clip input_ids to the last token when past exists.
        if past_key_values is not None and len(past_key_values) > 0:
            if input_ids is not None:
                past_length = past_key_values.get_seq_length()
                if cache_position is not None:
                    cache_position = cache_position[-1:]
                    if input_ids.shape[1] > 1:
                        input_ids = input_ids[:, -1:]
                elif past_length < input_ids.shape[1]:
                    input_ids = input_ids[:, past_length:]
                else:
                    input_ids = input_ids[:, -1:]

            if attention_mask is not None and cache_position is not None:
                attention_mask = attention_mask[:, -cache_position.shape[0] - 1 :]
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "image_hidden_states": image_hidden_states,
        }
        if logits_to_keep is not None:
            model_inputs["logits_to_keep"] = logits_to_keep

        # After the first iteration, images are already encoded.
        is_first = cache_position is None or cache_position[0].item() == 0
        if image_hidden_states is not None or not is_first:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_attention_mask"] = None

        return model_inputs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DopamineVLAVisionConfig",
    "DopamineVLAConfig",
    "DopamineVLAPreTrainedModel",
    "DopamineVLAVisionTransformer",
    "DopamineVLABaseModelOutputWithPast",
    "DopamineVLACausalLMOutputWithPast",
    "DopamineVLAConnector",
    "DopamineVLAModel",
    "DopamineVLAForConditionalGeneration",
]
