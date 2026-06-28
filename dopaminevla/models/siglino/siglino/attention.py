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
# Uses native SDPA (FlashAttention-2 on CUDA) with boolean mask support

import torch
import torch.nn.functional as F
from torch import nn

from .rope import apply_3d_rotary_emb


def create_sdpa_attention_mask(full_mask: torch.Tensor) -> torch.Tensor:
    """Convert a 2D padding mask (N, S) to a 4D SDPA boolean mask (N, 1, S, S).

    SDPA natively supports boolean masks where True = valid, False = padding.
    """
    valid_q = full_mask.unsqueeze(-1)
    valid_kv = full_mask.unsqueeze(-2)
    mask_matrix = valid_q & valid_kv
    return mask_matrix.unsqueeze(1)  # Return boolean mask directly


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
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = head_dim or dim // n_heads
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim

        super().__init__()

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

        # RoPE is element-wise on head_dim — apply before GQA expansion
        xq, xk = apply_3d_rotary_emb(xq, xk, freqs_cis, freqs_cis_2d, pos_thw)

        # SDPA handles GQA natively via enable_gqa, no need to repeat KV heads
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=attention_masks, enable_gqa=True
        )
        output = output.transpose(1, 2).contiguous().reshape(bs, seqlen, -1)

        return self.wo(output)


__all__ = ["Attention", "create_sdpa_attention_mask"]
