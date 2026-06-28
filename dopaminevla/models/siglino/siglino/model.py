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
#
# Edited by Gabriel Amaral

# Main model implementation for Falcon Vision Encoder
# A pure vision transformer distilled from DINOv3 and SigLIP2

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from .attention import Attention, create_sdpa_attention_mask
from .configs import SigLinoArgs
from .moe import FeedForward, MoE
from .rope import (
    apply_golden_freqs_cis_to_visual_pos,
    precompute_freqs_cis,
    precompute_golden_freqs_cis,
)


@dataclass
class SigLinoFeatures:
    features_siglip: torch.Tensor
    features_dinov3: torch.Tensor
    features_siglino: torch.Tensor
    grid_hw: tuple[int, int]


class Siglip2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = ACT2FN["gelu_pytorch_tanh"](hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling for SigLIP2-style summary features."""

    def __init__(self, hidden_size: int, num_attention_heads: int, output_dim: int) -> None:
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.attention = nn.MultiheadAttention(hidden_size, num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = Siglip2MLP(hidden_size, 4304)

    def forward(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        # key_padding_mask: True = padding (masked out). attention_mask: True = valid.
        if attention_mask is not None:
            attn = attention_mask.view(batch_size, -1)
            key_padding_mask = ~attn.bool()
        else:
            key_padding_mask = None

        hidden_state = self.attention(
            probe, hidden_state, hidden_state, key_padding_mask=key_padding_mask
        )[0]
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state[:, 0]


class Adapter(nn.Module):
    """Feature adapter for projecting to teacher dimensions."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def init_weights(self) -> None:
        w0 = self.net[0]
        w3 = self.net[3]
        b1 = self.net[1]
        assert isinstance(w0, nn.Linear)
        assert isinstance(w3, nn.Linear)
        assert isinstance(b1, nn.LayerNorm)
        nn.init.trunc_normal_(w0.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(w3.weight, mean=0.0, std=0.01)
        nn.init.zeros_(w0.bias)
        b1.reset_parameters()


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: SigLinoArgs) -> None:
        super().__init__()
        self.dim = args.dim
        self.parameterized_norm = args.parameterized_norm

        if args.parameterized_norm:
            self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
            self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

        self.attention = Attention(
            dim=args.dim,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            head_dim=args.head_dim,
            use_qk_norm=args.use_qk_norm,
            enable_3d_rope=args.enable_3d_rope,
            use_sink_attn=True,  # Match torchtitan checkpoint
        )

        # Dense FFN or MoE layer
        use_dense = layer_id < args.first_n_layers_dense
        if use_dense:
            ffn_hidden = args.ffn_dim if args.ffn_dim is not None else args.moe_dim
            self.feed_forward = FeedForward(args.dim, ffn_hidden, activation=args.activation)
            self.moe_enabled = False
        elif args.moe_args and args.moe_args.num_experts > 0:
            self.moe = MoE(args.moe_args, dim=args.dim, hidden_dim=args.moe_dim)
            self.moe_enabled = True
        else:
            self.feed_forward = FeedForward(args.dim, args.moe_dim)
            self.moe_enabled = False

        if args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        freqs_cis_2d: torch.Tensor | None = None,
        pos_thw: torch.Tensor | None = None,
        attention_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, D = x.shape
        if self.parameterized_norm:
            x_norm = self.attention_norm(x)
        else:
            x_norm = F.rms_norm(x, (x.size(-1),))
        h = x + self.attention(
            x_norm,
            freqs_cis,
            freqs_cis_2d,
            pos_thw,
            attention_masks=attention_masks,
        )

        if self.parameterized_norm:
            h_norm = self.ffn_norm(h)
        else:
            h_norm = F.rms_norm(h, (h.size(-1),))
        if self.moe_enabled:
            out = h + self.moe(h_norm)
        else:
            out = h + self.feed_forward(h_norm)

        return out

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            buffer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class SigLino(nn.Module):
    """
    SigLino - Agglomeration Mixture of Experts Vision Foundation Model
    """

    def __init__(self, args: SigLinoArgs) -> None:
        super().__init__()
        self.args = args
        self.n_layers = args.n_layers
        self.patch_size = args.spatial_patch_size
        self.n_storage_tokens = args.n_storage_tokens

        # Patch embedding: Conv2d fuses unfold + projection for raw images
        self.patch_embed = nn.Conv2d(
            in_channels=args.channel_size,
            out_channels=args.dim,
            kernel_size=args.spatial_patch_size,
            stride=args.spatial_patch_size,
            bias=False,
        )
        # Projection for pre-patched features (from processor output)
        n_pixels_per_patch = args.temporal_patch_size * args.spatial_patch_size**2
        self.img_projector = nn.Linear(
            n_pixels_per_patch * args.channel_size,
            args.dim,
            bias=False,
        )

        # CLS and register tokens
        self.cls_token = nn.Parameter(torch.empty(1, 1, args.dim))
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, self.n_storage_tokens, args.dim))

        # RoPE precomputed on CPU to survive meta-device init context from Transformers

        self.register_buffer("freqs_cis_golden", None)
        self.register_buffer("freqs_cis", None, persistent=False)

        # Transformer layers
        self.layers: nn.ModuleDict = nn.ModuleDict()
        for layer_id in range(args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, args)

        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

        # Teacher adapters
        self.teachers: dict[str, int] = dict(zip(args.teachers, args.teachers_dim))
        dinov3_dim = self.teachers.get("dinov3", 1280)
        siglip2_dim = self.teachers.get("siglip2", 1152)

        self.dinov3_adapter = Adapter(args.dim, dinov3_dim, bias=False)
        self.siglip2_adapter = Adapter(args.dim, siglip2_dim, bias=False)
        self.layer_norm_dinov3 = nn.LayerNorm(dinov3_dim)
        self.siglip2_multihead_attention_pooling_head = Siglip2MultiheadAttentionPoolingHead(
            siglip2_dim, 16, siglip2_dim
        )

        # Freeze teacher-specific components
        for param in self.layer_norm_dinov3.parameters():
            param.requires_grad = False
        for param in self.siglip2_multihead_attention_pooling_head.parameters():
            param.requires_grad = False

        # Block mask is created each call — caching is unsafe because the
        # mask-sum cache key is lossy for different mask patterns with same sum.

        # Precompute RoPE buffers on CPU (survives meta-device init context)
        self._post_init()

    def _precompute_freqs_cis(self, head_dim: int, args: SigLinoArgs) -> torch.Tensor:
        return precompute_freqs_cis(head_dim, args.max_seq_len, args.rope_theta)

    def _post_init(self) -> None:
        head_dim = self.args.head_dim or self.args.dim // self.args.n_heads
        d = head_dim // 2
        self.freqs_cis_golden = self._precompute_golden_freqs_cis(d, self.args)
        self.freqs_cis = self._precompute_freqs_cis(d, self.args)

    def _precompute_golden_freqs_cis(self, head_dim: int, args: SigLinoArgs) -> torch.Tensor:
        return precompute_golden_freqs_cis(
            args.n_heads, head_dim, args.rope_min_freqs, args.rope_max_freqs
        )

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        if self.freqs_cis is None:
            self._post_init()
        buffer_device = buffer_device or self.freqs_cis.device

        nn.init.trunc_normal_(self.patch_embed.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.img_projector.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)

        for layer in self.layers.values():
            assert isinstance(
                layer, TransformerBlock
            )  # Makes pyrefly happy and it doesnt yell at me
            layer.init_weights(buffer_device=buffer_device)

        self.norm.reset_parameters()
        self.dinov3_adapter.init_weights()
        self.siglip2_adapter.init_weights()

    @property
    def dtype(self) -> torch.dtype:
        return next(self.dinov3_adapter.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.dinov3_adapter.parameters()).device

    def _patchify(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert images to patches via Conv2d. Input: (N, C, H, W).

        If H or W are not divisible by ``patch_size``, the image is
        zero-padded to the next multiple so no trailing pixels are lost.
        """
        N, C, H, W = images.shape
        ph = pw = self.patch_size

        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_h > 0 or pad_w > 0:
            images = F.pad(images, (0, pad_w, 0, pad_h))
            H += pad_h
            W += pad_w

        h, w = H // ph, W // pw
        patches = self.patch_embed(images)  # (N, dim, h, w)
        patches = patches.flatten(2).transpose(1, 2)  # (N, h*w, dim)

        spatial_shape = torch.tensor([[h, w]] * N, device=images.device)
        return patches, spatial_shape

    def _build_vision_mask(self, full_mask: torch.Tensor) -> torch.Tensor:
        """Build SDPA attention mask from padding mask.

        Args:
            full_mask: (N, S) boolean mask where True = valid, False = padding

        Returns:
            4D attention mask tensor for SDPA
        """
        return create_sdpa_attention_mask(full_mask)

    def _get_thw_pos(
        self,
        batch_size: int,
        num_patches_per_image: int,
        spatial_shapes: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute position encodings for 2D golden RoPE.

        Patch-index approach: uses the sequence position j (0..L-1) to derive
        row = j // W, col = j %% W per image. Avoids any data-dependent tensor sizes
        (no torch.arange(max_H)).
        """
        N = batch_size
        R = 1 + self.n_storage_tokens  # CLS + registers
        S = R + num_patches_per_image  # Total sequence per image

        H_img = spatial_shapes[:, 0]  # (N,)
        W_img = spatial_shapes[:, 1]  # (N,)

        tpos = torch.zeros((N, S), dtype=torch.float32, device=device)
        hpos = torch.full((N, S), float("nan"), dtype=torch.float32, device=device)
        wpos = torch.full((N, S), float("nan"), dtype=torch.float32, device=device)

        # Patch index: (1, L) — size determined by input shape, not data values
        j = torch.arange(num_patches_per_image, device=device).unsqueeze(0)  # (1, L)

        # Per-image row/col: (N, L)
        h_idx = j // W_img.unsqueeze(-1)  # (N, L)
        w_idx = j % W_img.unsqueeze(-1)  # (N, L)

        # Per-image normalization factors: (N, 1)
        H_f = H_img.float().clamp(min=1).unsqueeze(-1)
        W_f = W_img.float().clamp(min=1).unsqueeze(-1)
        ylim = (H_f / W_f).sqrt()
        xlim = (W_f / H_f).sqrt()

        h_denom = (H_f - 1).clamp(min=1)
        w_denom = (W_f - 1).clamp(min=1)

        # Normalized coords per patch: (N, L)
        h_norm = -ylim + 2 * ylim * h_idx.float() / h_denom
        w_norm = -xlim + 2 * xlim * w_idx.float() / w_denom

        # Validity mask: patches whose row/col exceed actual image dims
        valid_mask = (h_idx < H_img.unsqueeze(-1)) & (w_idx < W_img.unsqueeze(-1))  # (N, L)

        # Write into sequence positions R..S-1, NaN for invalid patches
        hpos[:, R:] = torch.where(valid_mask, h_norm, float("nan"))
        wpos[:, R:] = torch.where(valid_mask, w_norm, float("nan"))

        return torch.stack(
            [tpos, hpos, wpos], dim=0
        )  # (3, N, S) — rearranged to (N, S, 3) in forward

    def forward(
        self,
        pixel_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, ...] | None]:
        """
        Forward pass for vision encoding.

        Args:
            pixel_values: Image patches (N, L, dim) - patches only, no CLS/register
            padding_mask: (N, L) mask where 1 = valid patch, 0 = padding
            spatial_shapes: Shape of each image (N, 2) with (H_patches, W_patches)

        Returns:
            Dictionary with:
            - "output": patch features {"dinov3": ..., "siglip2": ..., "siglino": ...}
            - "summary": pooled features {"dinov3": ..., "siglip2": ..., "siglino": ...}
        """

        # Handle raw images input
        if pixel_values.dim() == 4:
            # Save pre-padding mask and shapes before _patchify overwrites spatial_shapes
            _pre_padding_mask = padding_mask
            _pre_spatial_shapes = spatial_shapes

            pixel_values, spatial_shapes = self._patchify(pixel_values)
            N, L, _ = pixel_values.shape

            if _pre_padding_mask is not None and _pre_spatial_shapes is not None:
                # Pad mask from pre-padding grid to post-padding grid.
                # Pre-padding mask: (N, H_old*W_old) from _forward_branch
                # Post-padding spatial_shapes: (N, 2) with [H_new, W_new]
                h_old, w_old = (
                    _pre_spatial_shapes[0, 0].item(),
                    _pre_spatial_shapes[0, 1].item(),
                )
                h_new, w_new = (
                    spatial_shapes[0, 0].item(),
                    spatial_shapes[0, 1].item(),
                )
                old_mask = _pre_padding_mask.view(N, int(h_old), int(w_old))  # (N, H_old, W_old)
                pad_right = int(w_new - w_old)
                pad_bottom = int(h_new - h_old)
                if pad_right > 0 or pad_bottom > 0:
                    padding_mask = F.pad(old_mask, (0, pad_right, 0, pad_bottom), value=0.0)
                else:
                    padding_mask = old_mask
                padding_mask = padding_mask.reshape(N, -1).to(dtype=torch.float32)
            else:
                # No input mask — all patches are valid
                padding_mask = torch.ones((N, L), dtype=torch.float32, device=pixel_values.device)

        N, L, _ = pixel_values.shape
        device = pixel_values.device
        R = 1 + self.n_storage_tokens  # CLS + registers

        # Create default padding mask if not provided (all patches valid)
        if padding_mask is None:
            padding_mask = torch.ones((N, L), dtype=torch.float32, device=device)

        # Project patches from processor output to model dim (if not already projected)
        if pixel_values.shape[-1] != self.args.dim:
            h_NLD = self.img_projector(pixel_values)
        else:
            h_NLD = pixel_values

        # Add CLS and register tokens (these are always valid)
        cls_expanded = self.cls_token.expand(N, -1, -1)
        if self.n_storage_tokens > 0:
            reg_expanded = self.storage_tokens.expand(N, -1, -1)
            h_NSD = torch.cat([cls_expanded, reg_expanded, h_NLD], dim=1)
        else:
            h_NSD = torch.cat([cls_expanded, h_NLD], dim=1)

        S = h_NSD.shape[1]  # R + L

        # Build full mask: CLS+registers are always valid, then patch mask
        cls_reg_mask = torch.ones((N, R), dtype=padding_mask.dtype, device=device)
        full_mask = torch.cat([cls_reg_mask, padding_mask], dim=1)  # (N, S)
        full_mask_bool = full_mask.bool()

        # Build attention mask using padding mask
        block_mask = self._build_vision_mask(full_mask_bool)

        # Compute 2D RoPE positions
        assert spatial_shapes is not None, "spatial_shapes must be provided for 2D RoPE"
        thw_pos = self._get_thw_pos(N, L, spatial_shapes, device)
        pos_thw = thw_pos.permute(1, 2, 0).to(dtype=torch.float32)  # (3, N, S) -> (N, S, 3)

        # Mark CLS/register positions as NaN (no 2D RoPE for them)
        # Also mark padding positions as NaN
        patch_mask_2d = torch.zeros((N, S), dtype=torch.bool, device=device)
        patch_mask_2d[:, R:] = padding_mask.bool()  # Only valid patches get 2D RoPE
        pos_thw[:, :, 1:] = pos_thw[:, :, 1:].masked_fill(
            ~patch_mask_2d.unsqueeze(-1), float("nan")
        )

        freqs_cis_golden = apply_golden_freqs_cis_to_visual_pos(
            self.freqs_cis_golden.to(dtype=pos_thw.dtype), pos_thw[:, :, 1:]
        )

        # Transformer layers — collect intermediate hidden states when requested
        all_hidden_states = [] if output_hidden_states else None
        for layer in self.layers.values():
            h_NSD = layer(
                h_NSD,
                self.freqs_cis,
                freqs_cis_2d=freqs_cis_golden,
                pos_thw=pos_thw,
                attention_masks=block_mask,
            )
            if all_hidden_states is not None:
                all_hidden_states.append(h_NSD)

        h_NSD = self.norm(h_NSD)

        # Initialize unconditionally so the return below is always valid
        hidden_states_out: tuple[torch.Tensor, ...] | None = None
        if all_hidden_states is not None:
            # Apply final norm to intermediate states so they live in the same
            # representational space as the final output, then replace the last
            # entry with the already-normed final h_NSD to avoid double-norm.
            hidden_states_out = tuple(self.norm(hs) for hs in all_hidden_states[:-1]) + (h_NSD,)

        # Extract features
        cls_feats = h_NSD[:, 0]  # (N, D)
        patch_feats = h_NSD[:, R:]  # (N, L, D) - includes padding positions
        reg_and_patch_feats = h_NSD[:, 1:]  # (N, R-1+L, D) — registers + patches, no CLS

        # Project to teacher dimensions (patches only — teacher distillation)
        student_patch_dinov3 = self.dinov3_adapter(patch_feats)
        student_patch_siglip = self.siglip2_adapter(patch_feats)
        student_cls_dinov3 = self.dinov3_adapter(cls_feats)

        # SigLIP2 summary via attention pooling (uses full sequence with mask)
        h_sig = self.siglip2_adapter(h_NSD)
        # Pass the full mask for attention pooling
        siglip_attn_mask = full_mask  # (N, S) 2D mask for pooling head
        student_summary_siglip = self.siglip2_multihead_attention_pooling_head(
            h_sig, siglip_attn_mask
        )

        # Build mask for registers + patches (used by connector)
        reg_mask = torch.ones((N, R - 1), dtype=padding_mask.dtype, device=device)
        connector_mask = torch.cat([reg_mask, padding_mask], dim=1)  # (N, R-1+L)

        return {
            "patch_features": {
                "dinov3": student_patch_dinov3,
                "siglip2": student_patch_siglip,
                "siglino": reg_and_patch_feats,  # registers + patches for connector
            },
            "summary_features": {
                "dinov3": student_cls_dinov3,
                "siglip2": student_summary_siglip,
                "siglino": cls_feats,
            },
            "padding_mask": connector_mask,  # (N, R-1+L) float32 — 0 = padding, 1 = valid
            "hidden_states": hidden_states_out if output_hidden_states else None,
        }
