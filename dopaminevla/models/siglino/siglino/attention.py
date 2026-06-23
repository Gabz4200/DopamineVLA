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

# Attention module for Falcon Vision
# Supports FlexAttention (CUDA) or SDPA fallback (CPU) for device-agnostic attention

import einops as E
import torch
import torch.nn.functional as F
from torch import nn

from .rope import apply_3d_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match query heads."""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x.unsqueeze(3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# Lazy flex_attention import: only available on CUDA with compute capability >= 8.0
_HAS_FLEX_ATTENTION = False
_flex_attention = None
_AuxRequest = None
_BlockMask = None
_create_block_mask = None

try:
    from torch.nn.attention.flex_attention import (
        AuxRequest as _AuxRequest,
    )
    from torch.nn.attention.flex_attention import (
        BlockMask as _BlockMask,
    )
    from torch.nn.attention.flex_attention import (
        create_block_mask as _create_block_mask,
    )
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention,
    )

    _HAS_FLEX_ATTENTION = True
except ImportError:
    pass


def device_supports_flex_attention(device: torch.device) -> bool:
    """Check if the device supports flex_attention."""
    return _HAS_FLEX_ATTENTION and device.type == "cuda"


class FlexAttentionWrapper(nn.Module):
    """Wrapper for flex_attention with optional compilation and aux outputs.
    Falls back to SDPA on devices without flex_attention support."""

    _compiled = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: _BlockMask | None = None,
        compile: bool = True,
        return_aux: bool = False,
    ):
        fn = _flex_attention
        if compile and _flex_attention is not None:
            if FlexAttentionWrapper._compiled is None:
                FlexAttentionWrapper._compiled = torch.compile(
                    _flex_attention,
                    mode="max-autotune-no-cudagraphs",
                )
            fn = FlexAttentionWrapper._compiled

        if return_aux:
            return fn(q, k, v, block_mask=block_mask, return_aux=_AuxRequest(lse=True))
        return fn(q, k, v, block_mask=block_mask)


def create_sdpa_attention_mask(full_mask: torch.Tensor) -> torch.Tensor:
    """Convert a 2D padding mask (N, S) to a 4D SDPA attention mask (N, 1, S, S).
    Valid positions get 0.0, masked positions get -inf."""
    N, S = full_mask.shape
    valid_q = full_mask.unsqueeze(-1)
    valid_kv = full_mask.unsqueeze(-2)
    mask_matrix = valid_q & valid_kv
    attn_mask = torch.where(mask_matrix, 0.0, float("-inf"))
    attn_mask = attn_mask.unsqueeze(1)
    return attn_mask


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        use_qk_norm: bool = False,
        enable_3d_rope: bool = False,
        use_flex_attn: bool = True,
        use_sink_attn: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = head_dim or dim // n_heads
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim

        self.wq = nn.Linear(dim, self.q_dim, bias=False)
        self.wk = nn.Linear(dim, self.kv_dim, bias=False)
        self.wv = nn.Linear(dim, self.kv_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.use_qk_norm = use_qk_norm
        self.enable_3d_rope = enable_3d_rope
        self.use_flex_attn = use_flex_attn

        self.sink_attn = use_sink_attn
        if self.sink_attn:
            self.sinks = nn.Parameter(torch.empty(n_heads))

        self.inner_attention = FlexAttentionWrapper()

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.wo.weight)
        if self.sink_attn:
            nn.init.trunc_normal_(self.sinks, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        freqs_cis_2d: torch.Tensor | None = None,
        pos_thw: torch.Tensor | None = None,
        attention_masks: "_BlockMask | torch.Tensor | None" = None,
        compile: bool = True,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if self.use_qk_norm:
            xq = F.rms_norm(xq, (xq.size(-1),))
            xk = F.rms_norm(xk, (xk.size(-1),))

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq, xk = apply_3d_rotary_emb(xq, xk, freqs_cis, freqs_cis_2d, pos_thw)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        use_flex = self.use_flex_attn and isinstance(attention_masks, _BlockMask)

        if use_flex:
            output, aux = self.inner_attention(
                xq, xk, xv, block_mask=attention_masks, compile=compile, return_aux=True
            )
            sinks_BHL = E.rearrange(self.sinks, "h -> 1 h 1")
            sink_scale = torch.sigmoid(aux.lse - sinks_BHL)
            output = (output * sink_scale.unsqueeze(-1)).to(output.dtype)
            output = E.rearrange(output, "b h s d -> b s (h d)").contiguous()
        else:
            attn_mask = attention_masks if isinstance(attention_masks, torch.Tensor) else None
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask)
            output = output.transpose(1, 2).contiguous().reshape(bs, seqlen, -1)

        return self.wo(output)


def create_attention_mask(
    mask_mod,
    B: int | None,
    H: int | None,
    Q_LEN: int,
    KV_LEN: int,
    BLOCK_SIZE: tuple[int, int] = (64, 64),
) -> "_BlockMask":
    """Create a BlockMask for flex_attention."""
    return _create_block_mask(
        mask_mod,
        B=B,
        H=H,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        BLOCK_SIZE=BLOCK_SIZE,
    )
