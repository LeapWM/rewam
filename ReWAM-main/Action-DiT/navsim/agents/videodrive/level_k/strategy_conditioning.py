"""Ego Action Expert conditioning on surrounding-agent strategy tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .attention import CompatibleRMSNorm, MaskedMultiHeadAttention
from .types import StrategyTokenBatch


@dataclass
class EgoStrategyContext:
    """Other-agent tokens plus an empty-scene fallback token."""

    tokens: Tensor
    visibility_mask: Tensor
    has_valid_other: Tensor

    def validate(self) -> None:
        if self.tokens.ndim != 3:
            raise ValueError("tokens must have shape [B, K, D].")
        batch_size, num_tokens, _ = self.tokens.shape
        if num_tokens < 1:
            raise ValueError("EgoStrategyContext requires the fallback token.")
        if self.visibility_mask.shape != (batch_size, num_tokens):
            raise ValueError("visibility_mask must have shape [B, K].")
        if self.has_valid_other.shape != (batch_size,):
            raise ValueError("has_valid_other must have shape [B].")
        if not torch.is_floating_point(self.tokens):
            raise TypeError("tokens must use a floating-point dtype.")
        if self.visibility_mask.dtype != torch.bool:
            raise TypeError("visibility_mask must use torch.bool.")
        if self.has_valid_other.dtype != torch.bool:
            raise TypeError("has_valid_other must use torch.bool.")
        if self.visibility_mask.device != self.tokens.device:
            raise ValueError("Context tokens and masks must share one device.")
        if self.has_valid_other.device != self.tokens.device:
            raise ValueError("Context tokens and masks must share one device.")
        if not torch.all(self.visibility_mask.any(dim=-1)):
            raise ValueError("Every sample must expose at least one context token.")
        visible_tokens = self.tokens[self.visibility_mask]
        if not torch.isfinite(visible_tokens).all():
            raise ValueError("Visible Ego strategy context tokens must be finite.")


def prepare_ego_strategy_context(
    strategy_tokens: StrategyTokenBatch,
) -> EgoStrategyContext:
    """Remove the Ego token and add a zero fallback for empty agent sets."""

    strategy_tokens.validate()
    other_tokens = strategy_tokens.other_tokens
    other_mask = strategy_tokens.other_mask
    has_valid_other = other_mask.any(dim=-1)

    fallback_token = torch.zeros(
        (other_tokens.shape[0], 1, other_tokens.shape[-1]),
        device=other_tokens.device,
        dtype=other_tokens.dtype,
    )
    fallback_mask = (~has_valid_other)[:, None]
    context = EgoStrategyContext(
        tokens=torch.cat((other_tokens, fallback_token), dim=1),
        visibility_mask=torch.cat((other_mask, fallback_mask), dim=1),
        has_valid_other=has_valid_other,
    )
    context.validate()
    return context


class GatedStrategyConditioner(nn.Module):
    """Zero-initialized residual cross-attention for Ego Action tokens."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        context_dim: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        if context_dim != hidden_dim:
            raise ValueError(
                "Strategy context_dim must equal the Action hidden_dim."
            )
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.norm = CompatibleRMSNorm(hidden_dim)
        self.attention = MaskedMultiHeadAttention(
            query_dim=hidden_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        self.gate = nn.Parameter(torch.empty(hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Close the new residual branch at every scratch initialization."""

        nn.init.zeros_(self.gate)

    def forward(
        self,
        hidden_states: Tensor,
        context: EgoStrategyContext,
    ) -> Tensor:
        context.validate()
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B, T, D].")
        batch_size, num_steps, hidden_dim = hidden_states.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError("Unexpected Action hidden dimension.")
        if context.tokens.shape[0] != batch_size:
            raise ValueError("Action states and strategy context must share B.")
        if context.tokens.shape[-1] != self.context_dim:
            raise ValueError("Unexpected strategy context dimension.")
        if context.tokens.device != hidden_states.device:
            raise ValueError("Action states and strategy context must share device.")
        if context.tokens.dtype != hidden_states.dtype:
            raise TypeError("Action states and strategy context must share dtype.")
        if not torch.isfinite(hidden_states).all():
            raise ValueError("hidden_states must contain finite values.")

        visibility = context.visibility_mask[:, None].expand(
            batch_size,
            num_steps,
            context.tokens.shape[1],
        )
        strategy_delta = self.attention(
            self.norm(hidden_states),
            context.tokens,
            visibility,
        )
        strategy_delta = strategy_delta * context.has_valid_other[
            :, None, None
        ].to(dtype=strategy_delta.dtype)
        gated_delta = torch.tanh(self.gate)[None, None] * strategy_delta
        return hidden_states + gated_delta
