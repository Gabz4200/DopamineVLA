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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import einops as E
import torch
import torch.nn.functional as F
from torch import nn

from .attention import (
    Attention,
    BlockMask,
    create_attention_mask,
    create_sdpa_attention_mask,
    device_supports_flex_attention,
)
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


class PytorchGELUTanh(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh")


class Siglip2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.activation_fn = PytorchGELUTanh()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int | None = None) -> torch.Tensor:
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = torch.tensor(1.0, dtype=dtype) - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling for SigLIP2-style summary features."""

    def __init__(self, hidden_size: int, num_attention_heads: int, output_dim: int) -> None:
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.attention = nn.MultiheadAttention(hidden_size, num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = Siglip2MLP(hidden_size, 4304)
        self.num_heads = num_attention_heads

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        if attention_mask is not None:
            attention_mask = E.rearrange(attention_mask, "(b s) -> b s", b=batch_size)
            target_len, source_len = probe.shape[1], hidden_state.shape[1]
            attention_mask = _expand_mask(attention_mask, hidden_state.dtype, target_len)
            attention_mask = attention_mask.repeat(1, self.num_heads, target_len, 1)
            attention_mask = attention_mask.reshape(-1, target_len, source_len)

        hidden_state = self.attention(probe, hidden_state, hidden_state, attn_mask=attention_mask)[0]
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state[:, 0]


class Adapter(nn.Module):
    """Feature adapter for projecting to teacher dimensions."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def init_weights(self) -> None:
        nn.init.trunc_normal_(self.fc1.weight, mean=0.0, std=0.01)
        nn.init.trunc_normal_(self.fc2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        self.norm.reset_parameters()


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
            use_flex_attn=args.use_flex_attn,
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

        if args.depth_init if hasattr(args, "depth_init") else True:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        freqs_cis_2d: torch.Tensor | None = None,
        pos_thw: torch.Tensor | None = None,
        attention_masks: BlockMask | torch.Tensor | None = None,
        compile: bool = True,
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
            compile=compile,
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

        # Patch embedding
        self.n_pixels_per_patch = args.temporal_patch_size * args.spatial_patch_size**2
        self.img_projector = nn.Linear(
            self.n_pixels_per_patch * args.channel_size,
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
        self.siglip2_multihead_attention_pooling_head = Siglip2MultiheadAttentionPoolingHead(siglip2_dim, 16, siglip2_dim)

        # Freeze teacher-specific components
        for param in self.layer_norm_dinov3.parameters():
            param.requires_grad = False
        for param in self.siglip2_multihead_attention_pooling_head.parameters():
            param.requires_grad = False

        # Cache for block masks (instance-level, not shared across instances)
        self._cached_block_mask: BlockMask | None = None
        self._cached_mask_key: tuple[int, int, float, torch.device] | None = None

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
        return precompute_golden_freqs_cis(args.n_heads, head_dim, args.rope_min_freqs, args.rope_max_freqs)

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> Self:
        # Workaround to prevent casting complex RoPE buffers to real dtypes
        # (which triggers a warning and discards the imaginary part).

        # Identify complex buffers and remove them from standard application
        complex_buffers = {}

        # Iterate over a COPY of the items (or just keys) to avoid "dictionary changed size"
        for name, buf in list(self.named_buffers(recurse=False)):
            if buf is not None and buf.is_complex():
                complex_buffers[name] = buf
                del self._buffers[name]

        # Apply fn (device/dtype moves) to the rest of the model
        super()._apply(fn)

        # Handle complex buffers manually
        for name, buf in complex_buffers.items():
            # Probe fn to see if it performs a destructive cast to real
            dummy = torch.tensor([0.0], device=buf.device)
            res = fn(dummy)

            if not res.is_complex():
                # fn casts to real (e.g. bfloat16).
                # We should ONLY apply the device move, but keep the buffer complex.
                new_buf = buf.to(device=res.device)
            else:
                # fn preserves complex or is casting to complex. Safe to apply.
                new_buf = fn(buf)

            # Restore buffer with original persistence setting
            persistent = name not in self._non_persistent_buffers_set
            self.register_buffer(name, new_buf, persistent=persistent)

        return self

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        if self.freqs_cis is None:
            self._post_init()
        buffer_device = buffer_device or self.freqs_cis.device

        if self.img_projector is not None:
            nn.init.trunc_normal_(self.img_projector.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)

        for layer in self.layers.values():
            assert isinstance(layer, TransformerBlock)  # Makes pyrefly happy and it doesnt yell at me
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
        """Convert images to patches. Input: (N, C, H, W) or (N, H, W, C).

        If H or W are not divisible by ``patch_size``, the image is
        zero-padded to the next multiple so no trailing pixels are lost.
        """
        if images.shape[-1] == 3:  # NHWC format
            images = images.permute(0, 3, 1, 2)  # -> NCHW

        N, C, H, W = images.shape
        ph = pw = self.patch_size

        # Pad to patch_size multiple — no cropping, no info loss
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_h > 0 or pad_w > 0:
            images = F.pad(images, (0, pad_w, 0, pad_h))
            H += pad_h
            W += pad_w

        h, w = H // ph, W // pw

        # Create patches
        patches = images.unfold(2, ph, ph).unfold(3, pw, pw)
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # N, h, w, C, ph, pw
        patches = patches.reshape(N, h * w, C * ph * pw)

        # Create spatial shape tensor
        spatial_shape = torch.tensor([[h, w]] * N, device=images.device)

        return patches, spatial_shape

    def _use_flex_attn_on_device(self, device: torch.device) -> bool:
        return self.args.use_flex_attn and device_supports_flex_attention(device)

    def _build_vision_mask(
        self,
        full_mask: torch.Tensor,
        device: torch.device,
        block_size: int = 64,
    ) -> BlockMask | torch.Tensor:
        """Build attention mask using the padding mask.

        Args:
            full_mask: (N, S) boolean mask where True = valid, False = padding
            device: torch device
            block_size: FlexAttention block size (only used for flex path)

        Returns:
            _BlockMask for FlexAttention (CUDA) or 4D tensor mask (CPU)
        """
        N, S = full_mask.shape

        if self._use_flex_attn_on_device(device):
            mask_key = (N, S, full_mask.sum().item(), device)
            if self._cached_mask_key == mask_key and self._cached_block_mask is not None:
                return self._cached_block_mask

            valid_q = full_mask.unsqueeze(-1)
            valid_kv = full_mask.unsqueeze(-2)
            mask_matrix = valid_q & valid_kv

            def mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
                return mask_matrix[b, q_idx, kv_idx]

            block_mask = create_attention_mask(mask_mod, N, None, S, S, BLOCK_SIZE=(block_size, block_size))
            self._cached_block_mask = block_mask
            self._cached_mask_key = mask_key
            return block_mask
        else:
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

        return torch.stack([tpos, hpos, wpos], dim=0)  # (3, N, S) — rearranged to (N, S, 3) in forward

    def forward(
        self,
        pixel_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        compile: bool | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Forward pass for vision encoding.

        Args:
            pixel_values: Image patches (N, L, C*patch_size^2) - patches only, no CLS/register
            padding_mask: (N, L) mask where 1 = valid patch, 0 = padding
            spatial_shapes: Shape of each image (N, 2) with (H_patches, W_patches)
            compile: Whether to use compiled FlexAttention

        Returns:
            Dictionary with:
            - "output": patch features {"dinov3": ..., "siglip2": ..., "siglino": ...}
            - "summary": pooled features {"dinov3": ..., "siglip2": ..., "siglino": ...}
        """
        # Auto-detect compile: enabled on CUDA, disabled on CPU
        if compile is None:
            compile = pixel_values.device.type == "cuda"

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
                old_mask = _pre_padding_mask.view(N, h_old, w_old)  # (N, H_old, W_old)
                pad_right = w_new - w_old
                pad_bottom = h_new - h_old
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

        # Project patches
        h_NLD = self.img_projector(pixel_values)

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
        block_mask = self._build_vision_mask(full_mask_bool, device)

        # Compute 2D RoPE positions
        assert spatial_shapes is not None, "spatial_shapes must be provided for 2D RoPE"
        thw_pos = self._get_thw_pos(N, L, spatial_shapes, device)
        pos_thw = E.rearrange(thw_pos, "p n s -> n s p").to(dtype=torch.float32)

        # Mark CLS/register positions as NaN (no 2D RoPE for them)
        # Also mark padding positions as NaN
        patch_mask_2d = torch.zeros((N, S), dtype=torch.bool, device=device)
        patch_mask_2d[:, R:] = padding_mask.bool()  # Only valid patches get 2D RoPE
        pos_thw[:, :, 1:] = pos_thw[:, :, 1:].masked_fill(~patch_mask_2d.unsqueeze(-1), float("nan"))

        freqs_cis_golden = apply_golden_freqs_cis_to_visual_pos(self.freqs_cis_golden.to(dtype=pos_thw.dtype), pos_thw[:, :, 1:])

        # Transformer layers
        for layer in self.layers.values():
            h_NSD = layer(
                h_NSD,
                self.freqs_cis,
                freqs_cis_2d=freqs_cis_golden,
                pos_thw=pos_thw,
                attention_masks=block_mask,
                compile=compile,
            )

        h_NSD = self.norm(h_NSD)

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
        siglip_attn_mask = full_mask.reshape(-1)  # Flatten for pooling head
        student_summary_siglip = self.siglip2_multihead_attention_pooling_head(h_sig, siglip_attn_mask)

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
        }
