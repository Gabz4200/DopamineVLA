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

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, device: str = "cpu"
) -> torch.Tensor:
    """Precompute frequency tensor for 1D rotary embeddings (real-valued)."""
    _dev = device if device else DEVICE
    if end > 0:
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=_dev)[: (dim // 2)].float() / dim))
        t_cpu = torch.arange(end, device=_dev)
        freqs = torch.outer(t_cpu, freqs).float()
        # Return real-valued cos/sin instead of complex
        return torch.stack([freqs.cos(), freqs.sin()], dim=-1)
    else:
        return torch.tensor([], dtype=torch.float32, device=_dev)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input (HF-style real-valued RoPE)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    pos_t: torch.Tensor | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors (real-valued HF style)."""
    # freqs_cis shape: (S_max, D, 2) where D is frequencies per position
    # xq is (B, S, H, D) — need freqs to broadcast with S (dim 1) and D (dim 3)
    if freqs_cis.ndim == 3:
        if pos_t is not None:
            # Index select relevant positions, add head dim
            # (B, S, D, 2) -> (B, S, 1, D, 2)
            freqs_cis = freqs_cis[pos_t.long()].unsqueeze(2)
        else:
            # Use full sequence, add batch and head dims
            # (1, S, 1, D, 2)
            freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    elif pos_t is not None:
        # Legacy fallback for 2D freqs_cis
        freqs_cis = freqs_cis[pos_t.long()].unsqueeze(-3)

    # (B/S, S, 1, D, 2) -> freqs_cis[..., 0] = (B/S, S, 1, D)
    xq_cos = xq[..., : xq.shape[-1] // 2] * freqs_cis[..., 0]
    xq_sin = rotate_half(xq[..., : xq.shape[-1] // 2]) * freqs_cis[..., 1]
    xk_cos = xk[..., : xk.shape[-1] // 2] * freqs_cis[..., 0]
    xk_sin = rotate_half(xk[..., : xk.shape[-1] // 2]) * freqs_cis[..., 1]

    xq_out = torch.cat([xq_cos - xq_sin, xq[..., xq.shape[-1] // 2 :]], dim=-1)
    xk_out = torch.cat([xk_cos - xk_sin, xk[..., xk.shape[-1] // 2 :]], dim=-1)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_3d_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    freqs_cis_2d: torch.Tensor | None,
    pos_hw: torch.Tensor | None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 3D rotary embeddings (1D temporal + 2D spatial)."""
    xq_t, xq_hw = xq.chunk(chunks=2, dim=-1)
    xk_t, xk_hw = xk.chunk(chunks=2, dim=-1)

    pos_t = pos_hw[:, :, 0].clone() if pos_hw is not None else None
    xq_t, xk_t = apply_rotary_emb(xq_t, xk_t, freqs_cis, pos_t=pos_t)

    if freqs_cis_2d is not None and pos_hw is not None:
        xq_hw = apply_golden_rotary_emb(xq_hw, freqs_cis_2d, pos_hw[..., 1:])
        xk_hw = apply_golden_rotary_emb(xk_hw, freqs_cis_2d, pos_hw[..., 1:])

    xq_out = torch.cat([xq_t, xq_hw], dim=-1).type_as(xq)
    xk_out = torch.cat([xk_t, xk_hw], dim=-1).type_as(xk)
    return xq_out, xk_out


def _phi(m: int, device: str = "cpu") -> float:
    x = 2.0
    for _ in range(10):
        x = (1 + x) ** (1.0 / (m + 1.0))
    return x


def make_directions(n: int, d: int, device: str = "cpu") -> torch.Tensor:
    _dev = device if device else DEVICE
    m = d // 2
    phi_val = _phi(m)
    v1 = torch.linspace(1, phi_val, steps=n, device=_dev)
    z = 1 / (v1 + 1)
    directions = torch.stack([z * torch.cos(z), z * torch.sin(z)], dim=1)
    directions = torch.erfinv(2.0 * directions - 1.0)
    directions = directions / directions.norm(dim=1, keepdim=True)
    return directions.float()


def precompute_golden_freqs_cis(
    n_heads: int,
    head_dim: int,
    min_freq: float,
    max_freq: float,
    pos_dim: int = 2,
    p_zero_freqs: float = 0.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Precompute golden ratio based 2D frequencies for vision tokens (real-valued)."""
    n_freqs = head_dim // 2
    n_zero_freqs = round(p_zero_freqs * n_freqs)

    _dev = device if device else DEVICE
    zeros = torch.zeros(n_zero_freqs, device=_dev)

    if n_freqs - n_zero_freqs > 0:
        linspace_vals = torch.linspace(0, 1, n_freqs - n_zero_freqs, device=_dev)
        scaled_vals = min_freq * (max_freq / min_freq) ** linspace_vals
        omega_F = torch.cat((zeros, scaled_vals))
    else:
        omega_F = zeros

    directions_hFP = make_directions(n_heads * n_freqs, pos_dim).reshape(n_heads, n_freqs, pos_dim)
    return directions_hFP * omega_F.reshape(n_freqs, 1)


def apply_golden_freqs_cis_to_visual_pos(
    freqs_hFP: torch.Tensor, pos_BSP: torch.Tensor, device: str = "cpu"
) -> torch.Tensor:
    """Apply golden frequencies to visual positions (real-valued)."""
    img_mask_BS = ~(torch.isnan(pos_BSP).any(dim=-1))
    pos_tP = pos_BSP[img_mask_BS].float()
    theta_thF = torch.einsum("tp,hfp->thf", pos_tP, freqs_hFP.float())
    # Return real-valued (cos, sin) stacked
    return torch.stack([theta_thF.cos(), theta_thF.sin()], dim=-1)


def apply_golden_rotary_emb(
    input_BShd: torch.Tensor,
    freqs_cis_thF: torch.Tensor,
    pos_BSP: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Apply golden rotary embedding to image tokens only (real-valued HF style)."""
    img_mask_BS = ~(torch.isnan(pos_BSP).any(dim=-1))
    input_thd = input_BShd[img_mask_BS]

    # Input shape: (T, H, D) - need to split into two halves
    dim = input_thd.shape[-1]
    input_thd_1, input_thd_2 = input_thd[..., : dim // 2], input_thd[..., dim // 2 :]

    # Apply rotation: x * cos + rotate_half(x) * sin
    cos_thF = freqs_cis_thF[..., 0]
    sin_thF = freqs_cis_thF[..., 1]

    # (T, H, D1) * (T, D2) with proper broadcasting
    output_1 = input_thd_1 * cos_thF + rotate_half(input_thd_1) * sin_thF
    output_2 = input_thd_2 * cos_thF + rotate_half(input_thd_2) * sin_thF

    output_thd = torch.cat([output_1, output_2], dim=-1).type_as(input_BShd)

    # Scatter back to original positions
    img_mask_BS11 = img_mask_BS.unsqueeze(-1).unsqueeze(-1)
    return input_BShd.masked_scatter(img_mask_BS11.expand(input_BShd.shape), output_thd)


__all__ = [
    "precompute_freqs_cis",
    "precompute_golden_freqs_cis",
    "apply_golden_freqs_cis_to_visual_pos",
    "apply_3d_rotary_emb",
]
