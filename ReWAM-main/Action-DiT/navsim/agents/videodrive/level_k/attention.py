"""Pure-PyTorch attention primitives shared by Level-k experts."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CompatibleRMSNorm(nn.Module):
    """RMSNorm compatible with PyTorch versions lacking `nn.RMSNorm`."""

    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Restore the canonical identity scale for scratch initialization."""

        nn.init.ones_(self.weight)

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return normalized.to(dtype=hidden_states.dtype) * self.weight.to(
            dtype=hidden_states.dtype
        )


class MaskedMultiHeadAttention(nn.Module):
    """SDPA attention with an explicit boolean per-query visibility mask."""

    def __init__(
        self,
        *,
        query_dim: int,
        context_dim: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        inner_dim = num_heads * head_dim
        if query_dim <= 0 or context_dim <= 0:
            raise ValueError("query_dim and context_dim must be positive.")
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_heads and head_dim must be positive.")
        if inner_dim != query_dim:
            raise ValueError("num_heads * head_dim must equal query_dim.")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(query_dim, inner_dim)
        self.to_k = nn.Linear(context_dim, inner_dim)
        self.to_v = nn.Linear(context_dim, inner_dim)
        self.q_norm = CompatibleRMSNorm(inner_dim)
        self.k_norm = CompatibleRMSNorm(inner_dim)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(
        self,
        query_states: Tensor,
        context_states: Tensor,
        visibility_mask: Tensor | None = None,
    ) -> Tensor:
        if query_states.ndim != 3 or context_states.ndim != 3:
            raise ValueError(
                "query_states and context_states must have shape [B, L, D]."
            )
        batch_size, num_queries, _ = query_states.shape
        if context_states.shape[0] != batch_size:
            raise ValueError("Query and context must share batch size.")
        num_keys = context_states.shape[1]

        query = self.q_norm(self.to_q(query_states))
        key = self.k_norm(self.to_k(context_states))
        value = self.to_v(context_states)
        query = query.view(
            batch_size, num_queries, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, num_keys, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, num_keys, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attention_bias = None
        if visibility_mask is not None:
            if visibility_mask.shape != (batch_size, num_queries, num_keys):
                raise ValueError(
                    "visibility_mask must have shape [B, num_queries, num_keys]."
                )
            if visibility_mask.dtype != torch.bool:
                raise TypeError("visibility_mask must use torch.bool.")
            if visibility_mask.device != query_states.device:
                raise ValueError(
                    "visibility_mask and query_states must share one device."
                )
            if not torch.all(visibility_mask.any(dim=-1)):
                raise ValueError(
                    "Every attention query must see at least one context token."
                )
            attention_bias = torch.zeros(
                (batch_size, 1, num_queries, num_keys),
                device=query.device,
                dtype=query.dtype,
            )
            attention_bias.masked_fill_(
                ~visibility_mask[:, None],
                torch.finfo(query.dtype).min,
            )

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size, num_queries, -1
        )
        return self.to_out(attended)
