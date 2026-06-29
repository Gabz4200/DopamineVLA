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

"""Inputs merger — merges image hidden states into the token-embedding sequence."""

import torch
import torch.nn.functional as F
from torch import nn
from transformers.utils import torch_compilable_check


class DopamineVLAInputsMerger(nn.Module):
    """Merges image hidden states into the token-embedding sequence at image-token positions.

    ``image_hidden_states`` has shape ``(N_images, n_latents, text_hidden_size)``,
    where ``n_latents`` is the Perceiver's fixed token budget per image.
    """

    def __init__(self, image_token_id: int) -> None:
        super().__init__()
        self.image_token_id = image_token_id

    def forward(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        image_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("input_ids is required for image token merging")

        _, patch_size, _ = image_hidden_states.shape

        image_mask = input_ids == self.image_token_id

        num_image_tokens = image_mask.sum(dim=1)
        torch_compilable_check(
            torch.all(num_image_tokens % patch_size == 0),
            "At least one sample has <image> tokens not divisible by patch_size.",
        )
        blocks_per_sample = num_image_tokens // patch_size

        offsets = F.pad(blocks_per_sample.cumsum(dim=0), (1, 0), value=0)
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


__all__ = ["DopamineVLAInputsMerger"]
