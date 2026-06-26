# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
# Written by Orr Zohar
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
# Edited by Gabriel Amaral, 2026

"""
DopamineVLA bootstrapped from SmolVLM2 (transformers).

Provides DopamineVLA-prefixed classes mirroring the full SmolVLM2 architecture.
Each class either inherits from its SmolVLM counterpart (for PreTrainedModel
subclasses) or is a full standalone copy (for pure nn.Module components).

Future customizations: swap vision encoder to SigLino, modify connector, add action head.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationConfig
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.auto import AutoModel
from transformers.models.smolvlm.configuration_smolvlm import SmolVLMConfig, SmolVLMVisionConfig
from transformers.models.smolvlm.image_processing_smolvlm import SmolVLMImageProcessor
from transformers.models.smolvlm.modeling_smolvlm import (
    SmolVLMBaseModelOutputWithPast,
    SmolVLMCausalLMOutputWithPast,
    SmolVLMConnector,
    SmolVLMEncoder,
    SmolVLMForConditionalGeneration,
    SmolVLMModel,
    SmolVLMPreTrainedModel,
    SmolVLMSimpleMLP,
    SmolVLMVisionEmbeddings,
    SmolVLMVisionTransformer,
)
from transformers.models.smolvlm.processing_smolvlm import SmolVLMProcessor
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    can_return_tuple,
    logging,
    torch_compilable_check,
)
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

logger = logging.get_logger(__name__)


# ── Module-level helpers (copied from SmolVLM2) ──────────────────────────


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# ── Unique dataclasses (not from SmolVLM) ────────────────────────────────


@dataclass
class DopamineVLAVisionData:
    """Container for left/right vision observations processed for the policy."""

    original: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor


# ── Configuration ────────────────────────────────────────────────────────


class DopamineVLAVisionConfig(SmolVLMVisionConfig):
    """Vision configuration for DopamineVLA, inheriting from SmolVLMVisionConfig."""

    model_type = "dopaminevla_vision"


class DopamineVLAConfig(SmolVLMConfig):
    """Configuration class for DopamineVLA, inheriting from SmolVLMConfig."""

    model_type = "dopaminevla"


# ── Output dataclasses (inherit from SmolVLM counterparts) ───────────────


class DopamineVLABaseModelOutputWithPast(SmolVLMBaseModelOutputWithPast):
    """Base model output with past for DopamineVLA vision-language model."""

    pass


class DopamineVLACausalLMOutputWithPast(SmolVLMCausalLMOutputWithPast):
    """Causal LM output with past for DopamineVLA (includes action head logits in future)."""

    pass


# ── Vision subcomponents (inherit from SmolVLM for type compatibility) ──


class DopamineVLAVisionEmbeddings(SmolVLMVisionEmbeddings):
    """
    DopamineVLA vision embeddings, inheriting from SmolVLMVisionEmbeddings.
    All initialization and forward logic is handled by the parent class.
    Override here when custom patch embedding or position encoding is needed.
    """

    pass


class DopamineVLAVisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: SmolVLMVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).")
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Ignore copy
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input shape: Batch x Time x Channel"""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        queries = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        keys = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        values = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable[..., Any] = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation or "eager", eager_attention_forward)

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class DopamineVLAVisionMLP(nn.Module):
    def __init__(self, config: SmolVLMVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class DopamineVLAEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: SmolVLMVisionConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = DopamineVLAVisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = DopamineVLAVisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DopamineVLAEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers.
    Each layer is a [`DopamineVLAEncoderLayer`].
    """

    def __init__(self, config: SmolVLMVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([DopamineVLAEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> BaseModelOutput:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
            )

            hidden_states = layer_outputs

        return BaseModelOutput(last_hidden_state=cast(torch.FloatTensor, hidden_states))


# ── PreTrainedModel base (inherits from SmolVLM counterpart) ──────────────


class DopamineVLAPreTrainedModel(SmolVLMPreTrainedModel):
    """Base class for DopamineVLA models, inheriting from SmolVLMPreTrainedModel.

    Note: `config` type is inherited as `SmolVLMConfig`; at runtime the config
    object will be a `DopamineVLAConfig` (a subclass), so no type override is needed
    and no pyrefly bad-override-mutable-attribute error occurs.
    """

    # Update no-split module names to match our DopamineVLA-prefixed components
    _no_split_modules = ["DopamineVLAVisionAttention", "DopamineVLAEncoderLayer"]


# ── Vision Transformer (copies SmolVLMVisionTransformer form with DopamineVLA components) ──


class DopamineVLAVisionTransformer(SmolVLMVisionTransformer):
    """
    Vision Transformer for DopamineVLA, inheriting from SmolVLMVisionTransformer.

    Uses DopamineVLA-prefixed subcomponents so future changes to the vision
    stack (e.g. swapping to SigLino) can be made here. The config type is
    inherited as SmolVLMVisionConfig; at runtime a DopamineVLAVisionConfig may
    be used (it IS a SmolVLMVisionConfig).
    """

    _can_record_outputs = {
        "hidden_states": DopamineVLAEncoderLayer,
        "attentions": DopamineVLAVisionAttention,
    }

    def __init__(self, config: SmolVLMVisionConfig) -> None:
        # Call PreTrainedModel.__init__ directly — we want our own components,
        # not SmolVLM's. This avoids creating-and-discarding modules.
        PreTrainedModel.__init__(self, config)
        embed_dim = config.hidden_size

        self.embeddings = DopamineVLAVisionEmbeddings(config)
        self.encoder = cast(SmolVLMEncoder, DopamineVLAEncoder(config))
        self.patch_size = config.patch_size
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # post_init() calls init_weights(), initializing all our components.
        self.post_init()

    def get_input_embeddings(self) -> SmolVLMVisionEmbeddings:
        return self.embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embeddings = cast(SmolVLMVisionEmbeddings, value)

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_size = self.patch_size
            attn_mask = torch.ones(
                (
                    batch_size,
                    pixel_values.size(2) // patch_size,
                    pixel_values.size(3) // patch_size,
                )
            )
            attn_mask = attn_mask.to(dtype=torch.bool, device=pixel_values.device)
        else:
            attn_mask = patch_attention_mask

        hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=attn_mask)

        attn_mask = attn_mask.view(batch_size, -1)
        # Create the correct attention mask based on the attention implementation
        attn_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attn_mask,
        )

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attn_mask,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
        )


# ── Connector (inherits from SmolVLM counterparts for type compatibility) ──


class DopamineVLASimpleMLP(SmolVLMSimpleMLP):
    """Simple MLP projector for DopamineVLA, inheriting from SmolVLMSimpleMLP.

    Fully inherits behavior — override here when the connector projection needs changing.
    """

    pass


class DopamineVLAConnector(SmolVLMConnector):
    """
    Connector for DopamineVLA, inheriting from SmolVLMConnector.

    Projects vision encoder outputs into the text model embedding space.
    Skips parent __init__ to avoid creating-and-discarding SmolVLMSimpleMLP.
    Uses DopamineVLASimpleMLP so future projection changes are isolated.
    """

    def __init__(self, config: SmolVLMConfig) -> None:
        nn.Module.__init__(self)
        self.scale_factor = config.scale_factor
        self.modality_projection = DopamineVLASimpleMLP(config)


# ── Main Model (full code, uses DopamineVLA subcomponents) ────────────────


class DopamineVLAModel(SmolVLMModel):
    """
    DopamineVLA model consisting of a vision encoder and language decoder.

    Uses DopamineVLA-prefixed subcomponents. The vision_model can later be swapped
    to SigLino, and the connector can be customized for the action head.
    """

    def __init__(self, config: SmolVLMConfig) -> None:
        # Call PreTrainedModel.__init__ directly — we want full control over
        # component creation with our DopamineVLA classes.
        PreTrainedModel.__init__(self, config)
        self.padding_idx = self.config.text_config.pad_token_id
        self.vocab_size = self.config.text_config.vocab_size

        self.vision_model = DopamineVLAVisionTransformer._from_config(config.vision_config)
        self.connector = DopamineVLAConnector(config)
        self.text_model = AutoModel.from_config(config.text_config)

        self.image_seq_len = int(((config.vision_config.image_size // config.vision_config.patch_size) ** 2) / (config.scale_factor**2))
        self.image_token_id = self.config.image_token_id

        # post_init() initializes all submodule weights.
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.text_model.set_input_embeddings(value)

    def inputs_merger(self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor, image_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        This method aims at merging the token embeddings with the image hidden states into one single sequence of
        vectors that are fed to the transformer LM.
        """
        _, patch_size, _ = image_hidden_states.shape

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device))
            image_mask = image_mask[..., 0]  # slice off the hidden dim
        else:
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
        image_embeds[image_mask] = image_hidden_states[block_idx[image_mask], local_idx[image_mask], :]

        merged_embeds = torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds)
        return merged_embeds

    @can_return_tuple
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, ...] | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        pixel_attention_mask (`torch.Tensor`, *optional*):
            The attention mask indicating padded regions in the image.
        """
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.to(dtype=self.dtype)  # fp16 compatibility
        pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

        # Remove padding images - padding images are full 0.
        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image

        # If no images, leave one empty image.
        real_images_inds[0] |= ~torch.any(real_images_inds)

        pixel_values = pixel_values[real_images_inds].contiguous()
        # Handle the vision attention mask
        if pixel_attention_mask is None:
            attn_mask = torch.ones(
                size=[pixel_values.shape[i] for i in (0, 2, 3)],
                dtype=torch.bool,
                device=pixel_values.device,
            )
        else:
            # Remove padding images from the mask
            attn_mask = pixel_attention_mask.view(batch_size * num_images, *pixel_attention_mask.shape[2:])
            attn_mask = attn_mask[real_images_inds].contiguous()
        patch_size = self.config.vision_config.patch_size
        patches_subgrid = attn_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

        # Get sequence from the vision encoder
        image_outputs = self.vision_model(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask, return_dict=True, **kwargs)
        image_hidden_states = image_outputs.last_hidden_state

        # Modality projection & resampling
        image_features = self.connector(image_hidden_states)
        image_outputs.pooler_output = image_features

        return image_outputs

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
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            assert input_ids is not None  # validated above
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(input_ids.device)

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")

        if pixel_values is not None:
            image_hidden_states = self.get_image_features(pixel_values, pixel_attention_mask, return_dict=True).pooler_output
            image_hidden_states = image_hidden_states.to(inputs_embeds.device)
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=inputs_embeds.device)

        if image_hidden_states is not None:
            # When we generate, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist
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
            image_hidden_states=image_hidden_states,  # pyrefly: ignore[bad-argument-type]  # upstream: field is FloatTensor at runtime, not tuple[FloatTensor]
        )


# ── Conditional Generation (full code, uses DopamineVLA subcomponents) ────


class DopamineVLAForConditionalGeneration(SmolVLMForConditionalGeneration):
    """
    DopamineVLA model with a language modeling head.

    Composed of a vision encoder (to be replaced with SigLino), connector,
    language model, and future action head.
    """

    _tied_weights_keys = {"lm_head.weight": "model.text_model.embed_tokens.weight"}

    def __init__(self, config: SmolVLMConfig) -> None:
        # Call SmolVLMPreTrainedModel.__init__ (via its MRO) + GenerationMixin.
        # Skip SmolVLMForConditionalGeneration.__init__ so we can use our own
        # DopamineVLAModel from the start.
        SmolVLMPreTrainedModel.__init__(self, config)
        self.model = DopamineVLAModel(config)
        self.image_token_id = self.config.image_token_id
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.vocab_size = config.text_config.vocab_size
        self.model.text_model.generation_config = GenerationConfig.from_model_config(config)

        # Initialize weights and apply final processing
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
    ) -> tuple[torch.Tensor, ...] | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        pixel_attention_mask (`torch.Tensor`, *optional*):
            The attention mask indicating padded regions in the image.
        """
        return self.model.get_image_features(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, **kwargs)

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
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or `model.image_token_id`. Tokens with indices set to `model.image_token_id` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs)

        return DopamineVLACausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

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
        is_first_iteration: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Overwritten -- there are mutually exclusive inputs (if the logic to make `image_hidden_states` take
        # precedence is moved to the model, we can remove this fn)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if image_hidden_states is not None or not is_first_iteration:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_attention_mask"] = None

        return model_inputs


# ── Image Processor (inherits from SmolVLM counterpart) ───────────────────


class DopamineVLAImageProcessor(SmolVLMImageProcessor):
    """Image processor for DopamineVLA, inheriting from SmolVLMImageProcessor."""

    pass


# ── Processor (inherits from SmolVLM counterpart) ─────────────────────────


class DopamineVLAProcessor(SmolVLMProcessor):
    """Processor for DopamineVLA, inheriting from SmolVLMProcessor."""

    pass


# ── Public API ────────────────────────────────────────────────────────────


__all__ = [
    "DopamineVLAVisionConfig",
    "DopamineVLAConfig",
    "DopamineVLAPreTrainedModel",
    "DopamineVLAVisionData",
    "DopamineVLAVisionEmbeddings",
    "DopamineVLAVisionAttention",
    "DopamineVLAVisionMLP",
    "DopamineVLAEncoderLayer",
    "DopamineVLAEncoder",
    "DopamineVLAVisionTransformer",
    "DopamineVLABaseModelOutputWithPast",
    "DopamineVLACausalLMOutputWithPast",
    "DopamineVLASimpleMLP",
    "DopamineVLAConnector",
    "DopamineVLAModel",
    "DopamineVLAForConditionalGeneration",
    "DopamineVLAImageProcessor",
    "DopamineVLAProcessor",
]
