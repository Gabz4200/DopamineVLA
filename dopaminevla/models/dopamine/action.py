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

        # Linear component (w0, b0) Outputs 1 feature
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(1))

        # Periodic component (w, b) Outputs (out_features - 1) features
        self.w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(out_features - 1))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        tau: Expected shape (batch_size, seq_len, in_features)
        """
        # Calculate periodic features using Sine
        v1 = torch.sin(torch.matmul(tau, self.w) + self.b)

        # Calculate linear features
        v2 = torch.matmul(tau, self.w0) + self.b0

        # Concatenate along the last dimension to achieve 'out_features'
        return torch.cat([v1, v2], dim=-1)


class ActionHead(nn.Module):
    def __init__(self, hidden_size: int = 768, num_actions: int = 16, num_heads: int = 8) -> None:
        super().__init__()

        # Initialize our custom Time2Vec module (1 input feature -> 64 output features)
        self.time2vec = Time2Vec(in_features=1, out_features=64)

        # Project the 64 Time2Vec features up to the required hidden_size (768)
        self.time_proj = nn.Linear(64, hidden_size)

        # Cross-Attention layer (Batteries included)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )

    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, hidden_size) - Visual/State representation
        times: (batch_size, num_actions, 1) - Action time fractions
        """
        # Generate the continuous time representations
        t_embeds = self.time2vec(times)

        # Bridge the dimensionality gap
        t_embeds = self.time_proj(t_embeds)

        # Cross-Attention
        # Query: Time tells the network what fractions of a second to look for
        # Key/Value: The state representation provides the context
        attn_output, _ = self.cross_attention(query=t_embeds, key=x, value=x)

        return attn_output


__all__ = ["ActionHead", "Time2Vec"]
