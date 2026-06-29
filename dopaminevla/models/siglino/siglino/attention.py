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
#
# Sink attention requires the per-(head, position) log-sum-exp (LSE) of the
# attention score matrix — a value not exposed by SDPA or MHA. We therefore
# use a single manual QK^T pass that yields both LSE and the softmax weights
# in one traversal. torch.compile fuses the matmul→logsumexp→softmax→matmul
# chain into efficient tiled kernels on both CUDA and CPU.

import torch
import torch.nn.functional as F
from torch import nn

from .rope import apply_3d_rotary_emb


def create_sdpa_attention_mask(full_mask: torch.Tensor) -> torch.Tensor:
    """Convert a 2D padding mask (N, S) to a 4D boolean mask (N, 1, S, S).

    Returns a boolean mask where True = valid, False = padding.
    """
    valid_q = full_mask.unsqueeze(-1)
    valid_kv = full_mask.unsqueeze(-2)
    return (valid_q & valid_kv).unsqueeze(1)  # (N, 1, S, S)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value heads to match query head count (GQA)."""
    if n_rep == 1:
        return x
    bs, slen, n_kv_heads, head_dim = x.shape
    return (
        x.unsqueeze(3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    n_heads: int
    n_kv_heads: int
    n_rep: int
    head_dim: int
    q_dim: int
    kv_dim: int

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        use_qk_norm: bool = False,
        enable_3d_rope: bool = False,
        use_sink_attn: bool = True,
    ) -> None:
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

        self.sink_attn = use_sink_attn
        if self.sink_attn:
            self.sinks = nn.Parameter(torch.empty(n_heads))

    def init_weights(self, init_std: float) -> None:
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
        attention_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if self.use_qk_norm:
            xq = F.rms_norm(xq, (xq.size(-1),))
            xk = F.rms_norm(xk, (xk.size(-1),))

        # GQA expansion then RoPE (matches original repeat_kv ordering)
        xk = _repeat_kv(xk, self.n_rep)
        xv = _repeat_kv(xv, self.n_rep)
        xq, xk = apply_3d_rotary_emb(xq, xk, freqs_cis, freqs_cis_2d, pos_thw)

        xq = xq.transpose(1, 2)  # (B, H, S, D_head)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Single QK^T pass — produces scores, LSE, and output in one traversal.
        # torch.compile fuses this into efficient tiled kernels on CUDA and CPU.
        scale = self.head_dim**-0.5
        scores = torch.matmul(xq, xk.transpose(-2, -1)) * scale  # (B, H, S, S)
        if attention_masks is not None:
            scores = scores.masked_fill(~attention_masks, float("-inf"))

        # LSE in float32 for numerical stability; -inf for fully-masked rows
        # → sigmoid(-inf − sinks) = 0, so those positions are zeroed out cleanly.
        lse = torch.logsumexp(scores.float(), dim=-1)  # (B, H, S)

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        # Fully-masked rows produce NaN via 0/0 in softmax; replace with 0.
        # The sink gate zeros the corresponding output regardless, but NaN * 0
        # propagates in IEEE 754 so we must clean it here.
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        output = torch.matmul(attn_weights.to(xv.dtype), xv)  # (B, H, S, D_head)

        # Sink attention: learned per-head threshold on LSE gates each position.
        # Matches original: sigmoid(lse − sinks); sinks (H,) → (1, H, 1).
        sink_scale = torch.sigmoid(lse - self.sinks.view(1, -1, 1))
        output = (output * sink_scale.unsqueeze(-1)).to(output.dtype)

        output = output.transpose(1, 2).contiguous().reshape(bs, seqlen, -1)
        return self.wo(output)


__all__ = ["Attention", "create_sdpa_attention_mask"]
