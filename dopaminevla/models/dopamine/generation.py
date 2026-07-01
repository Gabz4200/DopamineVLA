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

"""DopamineVLA conditional generation — LM head + action head + GenerationMixin."""

from typing import Any, Unpack

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.utils import TransformersKwargs, can_return_tuple

from .action import ActionHead
from .base import DopamineVLAPreTrainedModel
from .configuration_dopaminevla import DopamineVLAConfig
from .model import DopamineVLAModel
from .outputs import DopamineVLACausalLMOutputWithPast


class DopamineVLAForConditionalGeneration(DopamineVLAPreTrainedModel, GenerationMixin):
    """DopamineVLA with a language modeling head and action head."""

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

        self.action_head = ActionHead(config)

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
        # Always request hidden states — action head needs all layer outputs.
        output_hidden_states = True
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

        # Run action head on all hidden states
        all_hidden = outputs.hidden_states
        delta_actions, action_state = self.action_head(
            all_hidden, use_cache=use_cache if use_cache else False
        )

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
            action=action_state,
            delta_actions=delta_actions,
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
        if past_key_values is not None and len(past_key_values) > 0 and input_ids is not None:
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
            "use_cache": True,
        }
        if logits_to_keep is not None:
            model_inputs["logits_to_keep"] = logits_to_keep

        # After the first iteration, images are already encoded.
        is_first = cache_position is None or cache_position[0].item() == 0
        if image_hidden_states is not None or not is_first:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_attention_mask"] = None

        return model_inputs


__all__ = ["DopamineVLAForConditionalGeneration"]
