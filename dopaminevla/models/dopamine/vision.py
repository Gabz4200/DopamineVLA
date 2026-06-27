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

"""Vision transformer — multi-view SigLino encoder wrapper."""

import torch

from dopaminevla.models.siglino.siglino.hf_integration import SigLinoHFModel

from .base import DopamineVLAPreTrainedModel
from .configuration_dopaminevla import DopamineVLAVisionConfig
from .crop import DopamineVLAMultiViewCrop


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
        self.crop = DopamineVLAMultiViewCrop(self.patch_size)
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
        return self.crop(pixel_values, padding_mask=padding_mask)

    def _forward_branch(
        self,
        pixel_values: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        features = out["patch_features"]["siglino"]
        mask = out.get("padding_mask")
        if mask is not None:
            mask = mask.bool()  # (N, L) float32 -> bool
        else:
            N, L = features.shape[:2]
            mask = torch.ones(N, L, dtype=torch.bool, device=features.device)
        return features, mask

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Returns (features_tuple, masks_tuple), one tensor per view.

        Each feature tensor is (B, L_i, hidden_size).
        Each mask tensor is (B, L_i) bool — True = valid patch.
        """
        pixel_views, mask_views = self.crop(
            pixel_values=pixel_values,
            padding_mask=patch_attention_mask,
        )

        results = [self._forward_branch(p, m) for p, m in zip(pixel_views, mask_views, strict=True)]
        features = tuple(r[0] for r in results)
        masks = tuple(r[1] for r in results)
        return features, masks


__all__ = ["DopamineVLAVisionTransformer"]
