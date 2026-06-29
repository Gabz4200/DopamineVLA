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

# RoPE (Rotary Position Embedding) implementation for Falcon Vision
# Includes 1D RoPE and 2D Golden RoPE for vision tokens.
#
# Rotation is implemented via complex multiplication (view_as_complex / polar /
# view_as_real), which exactly matches the original implementation and works
# identically on CPU and CUDA.  The interleaved-pair layout used here
# (pairs [x0,x1], [x2,x3], …) is required for weight compatibility.

import torch


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, device: str = "cpu"
) -> torch.Tensor:
    """Precompute 1D RoPE frequency tensor as complex exponentials.

    Returns a complex tensor of shape (end, dim // 2) where each entry
    e^{i*theta_k} encodes the rotation for frequency k at each position.
    """
    if end > 0:
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = torch.arange(end, device=device)
        freqs = torch.outer(t, freqs).float()
        return torch.polar(torch.ones_like(freqs), freqs)  # (end, dim//2) complex
    else:
        return torch.zeros(0, dtype=torch.complex64, device=device)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    pos_t: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 1D rotary embeddings via complex multiplication.

    Exactly matches the original implementation: reshapes input into
    interleaved pairs, multiplies by the complex frequency tensor, and
    reshapes back.  Works on CPU and CUDA.
    """
    # (B, S, H, D) -> (B, S, H, D/2) complex  [interleaved pairs]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freqs_cis: (S, D/2) complex
    if pos_t is not None:
        # pos_t: (B, S) -> index freqs -> (B, S, D/2) -> add head dim -> (B, S, 1, D/2)
        freqs = freqs_cis[pos_t.long()].unsqueeze(-2)
    else:
        # (S, D/2) -> (1, S, 1, D/2) to broadcast over batch and heads
        freqs = freqs_cis.unsqueeze(0).unsqueeze(-2)

    xq_out = torch.view_as_real(xq_ * freqs).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_3d_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    freqs_cis_2d: torch.Tensor | None,
    pos_hw: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 3D rotary embeddings: 1D temporal RoPE + 2D spatial Golden RoPE."""
    xq_t, xq_hw = xq.chunk(chunks=2, dim=-1)
    xk_t, xk_hw = xk.chunk(chunks=2, dim=-1)

    pos_t = pos_hw[:, :, 0] if pos_hw is not None else None
    xq_t, xk_t = apply_rotary_emb(xq_t, xk_t, freqs_cis, pos_t=pos_t)

    if freqs_cis_2d is not None and pos_hw is not None:
        xq_hw = apply_golden_rotary_emb(xq_hw, freqs_cis_2d, pos_hw[..., 1:])
        xk_hw = apply_golden_rotary_emb(xk_hw, freqs_cis_2d, pos_hw[..., 1:])

    xq_out = torch.cat([xq_t, xq_hw], dim=-1).type_as(xq)
    xk_out = torch.cat([xk_t, xk_hw], dim=-1).type_as(xk)
    return xq_out, xk_out


def _phi(m: int) -> float:
    x = 2.0
    for _ in range(10):
        x = (1 + x) ** (1.0 / (m + 1.0))
    return x


def make_directions(n: int, d: int, device: str = "cpu") -> torch.Tensor:
    """Create 2D directions using golden-ratio low-discrepancy sequence.

    Uses the plastic constant (phi(d)) to create quasi-random 2D directions
    via fractional parts of a geometric sequence, then maps through erfinv
    to approximate a normal distribution on the unit circle.
    """
    g = _phi(d)
    alpha = (1.0 / g) ** torch.arange(1, d + 1, dtype=torch.float64, device=device)
    i = torch.arange(1, n + 1, dtype=torch.float64, device=device).unsqueeze(1)
    z = torch.fmod(i * alpha, 1.0)
    directions = torch.erfinv(2.0 * z - 1.0)
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
    """Precompute golden ratio based 2D frequency directions for vision tokens.

    Returns a real tensor of shape (n_heads, head_dim // 2, pos_dim) — the
    direction × frequency product used to compute per-position phase angles.
    """
    n_freqs = head_dim // 2
    n_zero_freqs = round(p_zero_freqs * n_freqs)

    zeros = torch.zeros(n_zero_freqs, device=device)

    if n_freqs - n_zero_freqs > 0:
        linspace_vals = torch.linspace(0, 1, n_freqs - n_zero_freqs, device=device)
        scaled_vals = min_freq * (max_freq / min_freq) ** linspace_vals
        omega_F = torch.cat((zeros, scaled_vals))
    else:
        omega_F = zeros

    directions_hFP = make_directions(n_heads * n_freqs, pos_dim, device=device).reshape(
        n_heads, n_freqs, pos_dim
    )
    return directions_hFP * omega_F.reshape(n_freqs, 1)


def apply_golden_freqs_cis_to_visual_pos(
    freqs_hFP: torch.Tensor, pos_BSP: torch.Tensor
) -> torch.Tensor:
    """Compute complex golden RoPE frequencies for the valid visual positions.

    Returns a complex tensor of shape (T, H, F) — one complex rotation per
    (token, head, frequency) triplet, where T = number of valid (non-NaN) tokens.
    """
    img_mask_BS = (~torch.isnan(pos_BSP)).all(dim=-1)
    pos_tP = pos_BSP[img_mask_BS].float()
    theta_thF = torch.einsum("tp,hfp->thf", pos_tP, freqs_hFP.float())
    return torch.polar(torch.ones_like(theta_thF), theta_thF)  # (T, H, F) complex


def apply_golden_rotary_emb(
    input_BShd: torch.Tensor,
    freqs_cis_thF: torch.Tensor,
    pos_BSP: torch.Tensor,
) -> torch.Tensor:
    """Apply golden rotary embedding to image tokens only via complex multiplication.

    Exactly matches the original: gathers valid (non-NaN) tokens, converts to
    interleaved-pair complex, multiplies by the complex frequency tensor, and
    scatters the result back.  Works on CPU and CUDA.
    """
    img_mask_BS = (~torch.isnan(pos_BSP)).all(dim=-1)
    input_thd = input_BShd[img_mask_BS]  # (T, H, D)

    # (T, H, D) -> (T, H, D/2) complex via interleaved pairs
    input_thd_c = torch.view_as_complex(input_thd.float().reshape(*input_thd.shape[:-1], -1, 2))
    output_thd = torch.view_as_real(input_thd_c * freqs_cis_thF).flatten(-2).type_as(input_BShd)

    img_mask_BS11 = img_mask_BS.unsqueeze(-1).unsqueeze(-1)
    return input_BShd.masked_scatter(img_mask_BS11.expand(input_BShd.shape), output_thd)


__all__ = [
    "precompute_freqs_cis",
    "precompute_golden_freqs_cis",
    "apply_golden_freqs_cis_to_visual_pos",
    "apply_3d_rotary_emb",
]
