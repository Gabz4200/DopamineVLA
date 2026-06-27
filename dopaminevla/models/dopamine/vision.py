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

"""Vision transformer — single-pass SigLino encoder wrapper.

No multi-view cropping: the SigLino encoder is resolution-independent and can
process images at any size in a single forward pass.
"""

import torch

from dopaminevla.models.siglino.siglino.hf_integration import SigLinoHFModel

from .base import DopamineVLAPreTrainedModel
from .configuration_dopaminevla import DopamineVLAVisionConfig


class DopamineVLAVisionTransformer(DopamineVLAPreTrainedModel):
    """Single-pass vision encoder wrapping SigLinoHFModel.

    Accepts arbitrary-resolution images and returns patch-level features for
    the Perceiver connector to compress into fixed-length latents.
    """

    def __init__(self, config: DopamineVLAVisionConfig) -> None:
        super().__init__(config)
        self.vision_model = SigLinoHFModel(config)
        self.patch_size = config.spatial_patch_size
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward a batch of images through the SigLino encoder.

        Parameters
        ----------
        pixel_values : (B, 3, H, W) torch.Tensor
            Normalised image tensor.
        patch_attention_mask : (B, H//patch_size, W//patch_size) bool or None
            Pixel-level attention mask converted to patch grid.
            When ``None``, a default all-valid mask is created so that
            SigLino's internal ``_patchify`` padding is correctly masked out.

        Returns
        -------
        features : (B, L, hidden_size) torch.Tensor
            Patch-level features (registers + patches, no CLS).
        mask : (B, L) torch.Tensor bool
            ``True`` = valid patch / register.
        """
        if patch_attention_mask is None:
            b, _, h, w = pixel_values.shape
            patch_attention_mask = torch.ones(
                (b, h // self.patch_size, w // self.patch_size),
                dtype=torch.bool,
                device=pixel_values.device,
            )
        return self._forward_branch(pixel_values, padding_mask=patch_attention_mask)

    def _forward_branch(
        self,
        pixel_values: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = pixel_values.shape

        out = self.vision_model(
            pixel_values=pixel_values,
            padding_mask=padding_mask.reshape(b, -1) if padding_mask is not None else None,
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


__all__ = ["DopamineVLAVisionTransformer"]
