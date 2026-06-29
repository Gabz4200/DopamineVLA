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

# MoE (Mixture of Experts) implementation for Falcon Vision
# Simplified from torchtitan's MoE for standalone use

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MoEArgs:
    num_experts: int = 8
    num_shared_experts: int = 1
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0
    score_before_experts: bool = True
    top_k: int = 1
    activation: Literal["silu", "relu2"] = "silu"


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, activation: str = "silu") -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act == "relu2":
            return self.w2(2 * F.relu(self.w1(x)).square() * self.w3(x))
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float = 0.02) -> None:
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w3.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.w2.weight)


def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    act: str = "silu",
) -> torch.Tensor:
    num_tokens_list = num_tokens_per_expert.to(torch.int32).tolist()
    total_tokens = sum(num_tokens_list)
    num_padding = x.shape[0] - total_tokens
    x_splits = torch.split(x[:total_tokens], split_size_or_sections=num_tokens_list, dim=0)

    # Pad and stack for single batched matmul launch
    stacked_x = torch.nn.utils.rnn.pad_sequence(
        list(x_splits), batch_first=True
    )  # (E, max_tokens, dim)

    if act == "relu2":
        h = 2 * F.relu(torch.bmm(stacked_x, w1.transpose(1, 2))).square()
    else:
        h = F.silu(torch.bmm(stacked_x, w1.transpose(1, 2)))
    h = h * torch.bmm(stacked_x, w3.transpose(1, 2))
    out = torch.bmm(h, w2.transpose(1, 2))  # (E, max_tokens, dim)

    # Mask out padding tokens and flatten
    max_tokens = stacked_x.shape[1]
    arange = torch.arange(max_tokens, device=x.device)
    mask = arange[None, :] < num_tokens_per_expert[:, None]  # (E, max_tokens)
    out = out[mask]  # (total_tokens, dim)
    if num_padding > 0:
        out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
    return out


class GroupedExperts(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int, num_experts: int, activation: str = "silu"
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.activation = activation

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        return _run_experts_for_loop(
            self.w1, self.w2, self.w3, x, num_tokens_per_expert, self.activation
        )

    def init_weights(self, init_std: float) -> None:
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.zeros_(self.w2)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


class TokenChoiceTopKRouter(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        score_func: str = "sigmoid",
        route_norm: bool = False,
        route_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self.gate(x)
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.float())
        else:
            scores = F.softmax(scores.float(), dim=1)

        routing_scores = scores + expert_bias if expert_bias is not None else scores
        _, selected_experts_indices = torch.topk(routing_scores, k=self.top_k, dim=1)

        top_scores = scores.gather(dim=1, index=selected_experts_indices)
        if self.route_norm:
            top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
        top_scores = top_scores * self.route_scale

        num_tokens_per_expert = torch.bincount(
            selected_experts_indices.view(-1), minlength=self.num_experts
        )
        return top_scores, selected_experts_indices, num_tokens_per_expert.to(torch.float32)

    def init_weights(self, init_std: float) -> None:
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


class MoE(nn.Module):
    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int) -> None:
        super().__init__()
        num_experts = moe_args.num_experts

        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            activation=moe_args.activation,
        )
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=moe_args.top_k,
            score_func=moe_args.score_func,
            route_norm=moe_args.route_norm,
            route_scale=moe_args.route_scale,
        )
        self.shared_experts = (
            FeedForward(
                dim=dim,
                hidden_dim=hidden_dim * moe_args.num_shared_experts,
                activation=moe_args.activation,
            )
            if moe_args.num_shared_experts > 0
            else None
        )
        self.score_before_experts = moe_args.score_before_experts
        self.top_k = moe_args.top_k

        # Register buffer for load balancing (matches torchtitan checkpoint)
        self.register_buffer(
            "expert_bias",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(
            x, expert_bias=self.expert_bias
        )

        # Reorder tokens by expert
        token_indices_sorted = torch.argsort(selected_experts_indices.view(-1), stable=True)
        top_scores_sorted = top_scores.view(-1)[token_indices_sorted]
        token_indices_sorted = token_indices_sorted // self.top_k

        token_indices_expanded = token_indices_sorted.view(-1, 1).expand(-1, dim)
        routed_input = torch.gather(x, dim=0, index=token_indices_expanded)

        if self.score_before_experts:
            routed_input = (routed_input.float() * top_scores_sorted.view(-1, 1)).to(x.dtype)

        routed_output = self.experts(routed_input, num_tokens_per_expert)

        if self.shared_experts is not None:
            out = self.shared_experts(x)
        else:
            out = torch.zeros_like(x)

        routed_output = (routed_output.to(torch.float32) * top_scores_sorted.view(-1, 1)).to(
            x.dtype
        )

        out = out.scatter_add(dim=0, index=token_indices_expanded, src=routed_output)
        return out.view(bs, slen, dim)

    def init_weights(self, init_std: float) -> None:
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_experts is not None:
            self.shared_experts.init_weights(init_std)
