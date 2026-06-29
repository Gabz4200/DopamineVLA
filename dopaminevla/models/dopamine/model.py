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

"""Core DopamineVLA model — vision encoder + connector + text decoder."""

from typing import Unpack

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.auto import AutoModel
from transformers.utils import TransformersKwargs, can_return_tuple, logging

from .base import DopamineVLAPreTrainedModel
from .configuration_dopaminevla import DopamineVLAConfig
from .connector import DopamineVLAConnector
from .merger import DopamineVLAInputsMerger
from .outputs import DopamineVLABaseModelOutputWithPast
from .vision import DopamineVLAVisionTransformer

logger = logging.get_logger(__name__)


class DopamineVLAModel(DopamineVLAPreTrainedModel):
    """DopamineVLA model: vision encoder + connector + text decoder."""

    def __init__(self, config: DopamineVLAConfig) -> None:
        super().__init__(config)
        self.vision_model: DopamineVLAVisionTransformer = DopamineVLAVisionTransformer._from_config(
            config.vision_config
        )
        self.connector = DopamineVLAConnector(config)
        self.text_model = AutoModel.from_config(config.text_config)

        self.inputs_merger = DopamineVLAInputsMerger(self.config.image_token_id)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.text_model.set_input_embeddings(value)

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
        batch_size, num_images, _, height, width = pixel_values.shape
        pixel_values = pixel_values.to(dtype=self.dtype)
        pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

        # Remove padding images (all-zero tensors).
        real_images_inds = pixel_values.any(dim=(-1, -2, -3))

        # Ensure at least one image per batch is kept (avoids empty selection).
        real_images_inds[0] |= ~torch.any(real_images_inds)

        pixel_values = pixel_values[real_images_inds].contiguous()

        if pixel_attention_mask is None:
            attn_mask = torch.ones(
                (batch_size * num_images, height, width),
                dtype=torch.bool,
                device=pixel_values.device,
            )
        else:
            attn_mask = pixel_attention_mask.view(batch_size * num_images, height, width)
        attn_mask = attn_mask[real_images_inds].contiguous()

        patch_size = self.config.vision_config.spatial_patch_size
        # Check if any pixel in a patch is valid using max pooling.
        patch_attention_mask = (
            F.max_pool2d(
                attn_mask.float().unsqueeze(1),
                kernel_size=patch_size,
                stride=patch_size,
            )
            .squeeze(1)
            .bool()
        )

        features, masks = self.vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        )
        image_features = self.connector(features, attention_masks=masks)
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

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            assert input_ids is not None  # guaranteed by the ValueError guard above
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(input_ids.device)

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


__all__ = ["DopamineVLAModel"]
