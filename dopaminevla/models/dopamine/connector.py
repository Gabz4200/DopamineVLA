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

"""Perceiver-based connector — fuses multi-view visual features into fixed-length tokens."""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .configuration_dopaminevla import DopamineVLAConfig


class DopamineVLAPerceiverAttention(nn.Module):
    """
    Cross-attention block: learnable latents (queries) attend to the
    concatenation of [context, latents] (keys/values).
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int | None = None,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.scale = head_dim**-0.5
        self.attn_dropout = attn_dropout

        self.q_proj = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        latents: torch.Tensor,  # (B, n_latents, hidden_size)
        context: torch.Tensor,  # (B, seq,       hidden_size)
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, q_len, _ = latents.shape
        kv_len = q_len + context.size(1)

        kv_input = torch.cat([context, latents], dim=1)  # (B, seq+n_latents, D)

        q = self.q_proj(latents)
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)

        q = q.view(B, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, kv_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, kv_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            scale=self.scale,
            enable_gqa=True,
        )

        attn_out = attn_out.transpose(1, 2).reshape(B, q_len, self.n_heads * self.head_dim)
        return self.o_proj(attn_out)


class DopamineVLAPerceiverLayer(nn.Module):
    """One Perceiver block: pre-norm cross-attn + pre-norm FFN + residuals."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int | None = None,
        ffn_mult: int = 4,
        attn_dropout: float = 0.0,
        rms_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.latents_norm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.context_norm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.cross_attn = DopamineVLAPerceiverAttention(
            hidden_size, n_heads, head_dim, n_kv_heads, attn_dropout
        )
        self.post_attn_norm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ffn_mult, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size * ffn_mult, hidden_size, bias=False),
        )

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latents = latents + self.cross_attn(
            self.latents_norm(latents),
            self.context_norm(context),
            attention_mask=attention_mask,
        )
        latents = latents + self.ffn(self.post_attn_norm(latents))
        return latents


class DopamineVLAConnector(nn.Module):
    """Perceiver connector — compresses patch-level features into fixed-length latents.

    Accepts one or more view tensors (from a single or multi-view vision
    encoder) and cross-attends learned ``n_latents`` queries into the
    concatenated context.  Output shape is always
    ``(B, n_latents, text_hidden_size)``, independent of input patch count.

    Parameters are read from the ``vision_connector_*`` fields of
    ``DopamineVLAConfig``.
    """

    def __init__(self, config: DopamineVLAConfig) -> None:
        super().__init__()

        self.n_latents = config.vision_connector_n_latents
        vision_hidden_size = config.vision_config.hidden_size
        text_hidden_size = config.text_config.hidden_size

        n_layers = config.vision_connector_n_layers
        n_heads = config.vision_connector_n_heads
        head_dim = config.vision_connector_head_dim
        n_kv_heads = config.vision_connector_n_kv_heads
        ffn_mult = config.vision_connector_ffn_mult
        attn_dropout = config.vision_connector_attn_dropout
        rms_eps = config.vision_connector_rms_eps

        # Learnable query embeddings
        self.latents = nn.Parameter(
            torch.empty(self.n_latents, vision_hidden_size).normal_(std=0.02)
        )

        self.layers = nn.ModuleList(
            [
                DopamineVLAPerceiverLayer(
                    hidden_size=vision_hidden_size,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    n_kv_heads=n_kv_heads,
                    ffn_mult=ffn_mult,
                    attn_dropout=attn_dropout,
                    rms_eps=rms_eps,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.RMSNorm(vision_hidden_size, eps=rms_eps)
        self.modality_projection = nn.Linear(vision_hidden_size, text_hidden_size, bias=False)

    def forward(
        self,
        view_hidden_states: Tuple[torch.Tensor, ...],
        attention_masks: Tuple[torch.Tensor | None, ...] | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        view_hidden_states : tuple of Tensor, each (B, L_i, vision_hidden_size)
            Vision-encoder outputs for each view.  L_i may differ across views.
        attention_masks : tuple of optional bool Tensor, each (B, L_i), optional
            Padding masks for each view (True = valid).  ``None`` = all valid.

        Returns
        -------
        Tensor shape ``(B, n_latents, text_hidden_size)``
        """
        B = view_hidden_states[0].size(0)

        # 1. Concatenate all views into one context sequence.
        context = (
            view_hidden_states[0]
            if len(view_hidden_states) == 1
            else torch.cat(view_hidden_states, dim=1)
        )  # (B, sum(L_i), D_vis)

        # 2. Build a unified attention mask for the full context + latents.
        attn_mask = None
        if attention_masks is not None:
            valid_parts: list[torch.Tensor] = []
            for i, m in enumerate(attention_masks):
                if m is None:
                    L_i = view_hidden_states[i].size(1)
                    valid_parts.append(torch.ones(B, L_i, dtype=torch.bool, device=context.device))
                else:
                    valid_parts.append(m)

            ctx_mask = torch.cat(valid_parts, dim=1)
            lat_mask = torch.ones(B, self.n_latents, dtype=torch.bool, device=context.device)
            full_mask = torch.cat([ctx_mask, lat_mask], dim=1)  # (B, kv_len)

            attn_mask = full_mask[:, None, None, :]  # (B, 1, 1, kv_len) bool, broadcasts over q_len

        # 3. Expand latents over batch (zero-copy view).
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, n_latents, D_vis)

        for layer in self.layers:
            latents = layer(latents, context, attention_mask=attn_mask)

        latents = self.norm(latents)
        latents = self.modality_projection(latents)  # (B, n_latents, D_text)
        return latents


__all__ = [
    "DopamineVLAPerceiverAttention",
    "DopamineVLAPerceiverLayer",
    "DopamineVLAConnector",
]
