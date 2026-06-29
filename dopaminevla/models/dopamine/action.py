import torch
import torch.nn as nn


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


class ActionHead(nn.Module):
    """Cross-attends time-encoded queries into the visual/state representation.

    Uses Time2Vec + linear projection to encode action times, then cross-attends
    those embeddings (as queries) into the state/visual context (as key/value).
    """

    def __init__(self, hidden_size: int = 768, num_heads: int = 8) -> None:
        super().__init__()

        self.time2vec = Time2Vec(in_features=1, out_features=64)
        self.time_proj = nn.Linear(64, hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )

    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, hidden_size) - Visual/State representation
        times: (batch_size, num_actions, 1) - Action time fractions
        """
        t_embeds = self.time_proj(self.time2vec(times))

        # Query: time fractions tell the network what to look for
        # Key/Value: state representation provides the context
        attn_output, _ = self.cross_attention(query=t_embeds, key=x, value=x)

        return attn_output


__all__ = ["ActionHead", "Time2Vec"]
