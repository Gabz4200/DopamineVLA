import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_dopaminevla import DopamineVLAConfig


class Time2Vec(nn.Module):
    """
    Time2Vec implementation based on the SineActivation logic.
    Computes a vector representation of time using a mix of linear and periodic functions.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.out_features = out_features

        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(out_features - 1))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        tau: Expected shape (batch_size, seq_len, in_features)
        """
        v1 = torch.sin(torch.matmul(tau, self.w) + self.b)
        v2 = torch.matmul(tau, self.w0) + self.b0
        return torch.cat([v1, v2], dim=-1)


class ActionSWABlock(nn.Module):
    """Sliding-window attention block with pre-norm FFN and residuals.

    Self-attention with a causal sliding window mask. Supports optional
    KV cache for incremental generation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int,
        ffn_mult: int = 4,
        rms_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, (
            "hidden_size must be divisible by num_heads"
        )
        self.window_size = window_size
        self.scale = self.head_dim**-0.5

        self.norm1 = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.norm2 = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ffn_mult, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size * ffn_mult, hidden_size, bias=False),
        )

    def _build_sliding_window_mask(
        self,
        new_len: int,
        full_len: int,
        new_start: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build causal sliding window mask.

        Returns (new_len, full_len) with 0.0 for allowed, -inf for masked.
        """
        rows = torch.arange(new_len, device=device, dtype=dtype).unsqueeze(1)  # (N, 1)
        cols = torch.arange(full_len, device=device, dtype=dtype).unsqueeze(0)  # (1, L)
        kv_pos = rows + new_start
        window_start = (kv_pos - self.window_size + 1).clamp(min=0)
        mask = cols - window_start
        mask = mask.to(dtype=dtype).where((mask >= 0) & (cols <= kv_pos), float("-inf"))
        return mask

    def forward(
        self,
        x: torch.Tensor,
        cache_k: torch.Tensor | None = None,
        cache_v: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, T_new, D) — input tokens.
            cache_k: (B, cached_len, H, head_dim) or None.
            cache_v: same shape as cache_k.
            use_cache: Whether to return updated KV cache.

        Returns:
            If use_cache: (output, (k_new, v_new))
            Otherwise: output only
            output shape: (B, T_new, D)
        """
        B, T_new, D = x.shape
        H = self.num_heads
        hd = self.head_dim

        # Pre-norm + QKV projection
        residual = x
        x = self.norm1(x)
        qkv = self.qkv_proj(x)  # (B, T_new, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T_new, D)

        # Reshape to (B, H, T, hd)
        q = q.view(B, T_new, H, hd).transpose(1, 2)
        k = k.view(B, T_new, H, hd).transpose(1, 2)
        v = v.view(B, T_new, H, hd).transpose(1, 2)

        if use_cache and cache_k is not None:
            # Concatenate with cached K, V
            k = torch.cat([cache_k, k], dim=2)
            v = torch.cat([cache_v, v], dim=2)

        full_len = k.shape[2]
        new_start = full_len - T_new

        # Build causal sliding window mask
        attn_mask = self._build_sliding_window_mask(
            T_new, full_len, new_start, device=x.device, dtype=q.dtype
        )

        # Attend
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)
        attn_out = attn_out.transpose(1, 2).reshape(B, T_new, D)
        attn_out = self.o_proj(attn_out)

        x = residual + attn_out

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        if use_cache:
            # Return updated cache (last K, V segments after concat)
            return x, (k, v)
        return x


class ActionHead(nn.Module):
    """Action prediction head with Time2Vec, cross-attention into hidden states,
    and sliding-window attention over accumulated action tokens.

    Operates in ``action_embed_dim`` (default 512) throughout. Only the final
    projection reduces to ``action_delta_dim`` (default 24).
    """

    def __init__(self, config: DopamineVLAConfig) -> None:
        super().__init__()

        self.num_queries = config.num_action_queries
        self.hidden_size = config.text_config.hidden_size
        self.action_embed_dim = config.action_embed_dim
        self.delta_dim = config.action_delta_dim

        # Time encoding: Time2Vec → projection to action_embed_dim
        self.time2vec = Time2Vec(in_features=1, out_features=64)
        self.time_proj = nn.Linear(64, self.action_embed_dim, bias=False)

        # Hidden state projection: hidden_size → action_embed_dim
        self.state_proj = nn.Linear(self.hidden_size, self.action_embed_dim, bias=False)

        # Cross-attention into hidden states
        self.cross_attn_norm = nn.RMSNorm(self.action_embed_dim, eps=1e-6)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.action_embed_dim,
            num_heads=config.action_cross_attention_heads,
            batch_first=True,
        )

        # SWA layers
        self.swa_norm = nn.RMSNorm(self.action_embed_dim, eps=1e-6)
        self.swa_layers = nn.ModuleList(
            [
                ActionSWABlock(
                    hidden_size=self.action_embed_dim,
                    num_heads=config.action_swa_heads,
                    window_size=config.action_swa_window_size,
                    ffn_mult=config.action_swa_ffn_mult,
                )
                for _ in range(config.action_swa_layers)
            ]
        )

        # Final projection to delta dimension
        self.delta_norm = nn.RMSNorm(self.action_embed_dim, eps=1e-6)
        self.delta_proj = nn.Linear(self.action_embed_dim, self.delta_dim, bias=False)

        # Learnable per-dimension fade for persistent action state
        self.action_fade = nn.Parameter(torch.zeros(self.delta_dim))

        # Persistent state (included in state_dict)
        self.register_buffer("_action_buffer", torch.empty(1, 0, self.action_embed_dim))
        self.register_buffer("_step_counter", torch.tensor(0, dtype=torch.long))
        self.register_buffer("persistent_action_state", torch.zeros(1, 1, self.delta_dim))

        # Non-persistent KV caches for SWA layers
        self._swa_kv_caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None

        # Non-zero init for all parameters to avoid stuck-at-zero gradients
        self._init_weights()

    def _init_weights(self) -> None:
        """Non-zero initialization for all parameters to avoid training stagnation at start."""
        # action_fade: break per-dimension symmetry so each starts with a different decay rate
        nn.init.trunc_normal_(self.action_fade, mean=0.0, std=0.25)

        # state_proj: MSE-style/xavier init for projecting hidden_size to action_embed_dim.
        # Default kaiming with fan_in=768 gives ~U(-0.062, 0.062) which squashes KV and
        # collapses cross-attention patterns. Xavier with gain=1.4 reserves more variance.
        nn.init.xavier_uniform_(self.state_proj.weight, gain=1.4)

        # time_proj: standard xavier for time embedding → action space
        nn.init.xavier_uniform_(self.time_proj.weight, gain=1.0)

        # delta_proj: output projection. Init with moderate variance so initial
        # deltas are non-trivial and the persistent state updates early.
        nn.init.xavier_uniform_(self.delta_proj.weight, gain=1.0)

    def reset_state(self) -> None:
        """Call before starting a new generation/episode."""
        self._action_buffer = torch.empty(
            1, 0, self.action_embed_dim, device=self._action_buffer.device
        )
        self._step_counter.zero_()
        self.persistent_action_state.zero_()
        self._swa_kv_caches = None

    def _compute_times(self, B: int, device: torch.device) -> torch.Tensor:
        """Return (B, N, 1) time fractions, one per query.

        Each forward pass covers one second split into N queries.
        Time value = (step * N + query_idx) / N  (continuous seconds from start).
        """
        step = self._step_counter.item()
        offsets = torch.arange(self.num_queries, device=device, dtype=torch.float)
        times = (offsets + step * self.num_queries) / self.num_queries
        return times.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)

    def forward(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run action head on text model hidden states.

        Args:
            hidden_states: Tuple of (B, S, D) tensors, one per text model layer
                (including embedding output).
            use_cache: Whether to use/update SWA KV caches.

        Returns:
            delta: (B, N, delta_dim) per-query deltas, tanh'd to [-1, 1] each.
            states: (B, N, delta_dim) N intermediate action states forming the
                1-second execution trajectory, one per 1/N s interval.
        """
        B = hidden_states[0].shape[0]

        # 1. Time encoding
        times = self._compute_times(B, device=hidden_states[0].device)
        t_embeds = self.time_proj(self.time2vec(times))  # (B, N, action_embed_dim)

        # 2. Project all hidden states to action_embed_dim and concatenate
        kv = self.state_proj(torch.cat(hidden_states, dim=1))  # (B, S_total, action_embed_dim)

        # 3. Cross-attention: time queries × hidden states
        t_embeds = self.cross_attn_norm(t_embeds)
        action_tokens, _ = self.cross_attention(query=t_embeds, key=kv, value=kv)

        # 4. Append to buffer (detach to prevent cross-step graph)
        with torch.no_grad():
            self._action_buffer = torch.cat([self._action_buffer, action_tokens.detach()], dim=1)
            window = self.swa_layers[0].window_size
            max_keep = max(window, self.num_queries * 2)
            if self._action_buffer.shape[1] > max_keep:
                self._action_buffer = self._action_buffer[:, -max_keep:, :]

        # 5. SWA layers
        is_incremental = use_cache and self._swa_kv_caches is not None
        swa_input = action_tokens if is_incremental else self._action_buffer
        swa_out = self.swa_norm(swa_input)

        new_caches: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for i, layer in enumerate(self.swa_layers):
            k, v = self._swa_kv_caches[i] if self._swa_kv_caches else (None, None)
            out = layer(swa_out, cache_k=k, cache_v=v, use_cache=use_cache)
            if use_cache:
                swa_out, (k_new, v_new) = out
                new_caches.append((k_new, v_new))
            else:
                swa_out = out
                new_caches.append(None)

        self._swa_kv_caches = new_caches if use_cache else None

        if not is_incremental:
            swa_out = swa_out[:, -self.num_queries :, :]

        # 6. Project to delta actions
        swa_out = self.delta_norm(swa_out)
        delta = self.delta_proj(swa_out)  # (B, N, delta_dim)

        # 7. Bound per-dimension to [-1, 1]
        delta = torch.tanh(delta)  # (B, N, delta_dim), each element in [-1, 1]

        # 8. Sequentially accumulate N intermediate states forming a 1-second trajectory.
        #    Formula: s_i = fade * (start_state + sum_{j<i} d_j) + d_i
        #    Controller executes these N states at 1/N-second intervals.
        start_state = self.persistent_action_state  # (B, 1, delta_dim)
        fade = torch.sigmoid(self.action_fade)  # (delta_dim,)
        cumulative = torch.zeros_like(start_state)  # running sum of prior deltas
        states: list[torch.Tensor] = []
        for i in range(self.num_queries):
            d = delta[:, i : i + 1, :]
            state = fade * (start_state + cumulative) + d
            states.append(state)
            cumulative = cumulative + d

        intermediate_states = torch.cat(states, dim=1)  # (B, N, delta_dim)
        self.persistent_action_state = state  # last intermediate state persists for next step

        # 9. Increment step counter
        self._step_counter += 1

        return delta, intermediate_states


__all__ = ["ActionHead", "Time2Vec", "ActionSWABlock"]
