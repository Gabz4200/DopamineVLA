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

"""Multi-view crop module — splits an image into three overlapping views."""

import torch
from torch import nn


class DopamineVLAMultiViewCrop(nn.Module):
    """Splits an image into three overlapping views: full, left crop, right crop.

    The crop boundaries are computed from patch-aligned coordinates so each
    view aligns with the vision encoder's patch grid.
    """

    def __init__(self, patch_size: int, overlap_pixels: int = 32) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.overlap_pixels = overlap_pixels

    def forward(
        self,
        pixel_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
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
        patch_overlap = min(self.overlap_pixels // self.patch_size, patch_mid)

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


__all__ = ["DopamineVLAMultiViewCrop"]
