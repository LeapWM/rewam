"""Normalization between physical and model-space surrounding-agent paths."""

from __future__ import annotations

import torch
from torch import Tensor


class OtherTrajectoryCodec:
    """Scale positions without clipping and normalize decoded headings."""

    def __init__(
        self,
        *,
        position_scale: float = 80.0,
        heading_epsilon: float = 1e-6,
    ) -> None:
        if position_scale <= 0:
            raise ValueError("position_scale must be positive.")
        if heading_epsilon <= 0:
            raise ValueError("heading_epsilon must be positive.")
        self.position_scale = position_scale
        self.heading_epsilon = heading_epsilon

    @staticmethod
    def _validate(
        trajectories: Tensor,
        agent_mask: Tensor | None,
    ) -> Tensor | None:
        if trajectories.ndim != 4 or trajectories.shape[-1] != 4:
            raise ValueError("trajectories must have shape [B, N, T, 4].")
        if not torch.is_floating_point(trajectories):
            raise TypeError("trajectories must use a floating-point dtype.")

        if agent_mask is None:
            if not torch.isfinite(trajectories).all():
                raise ValueError("trajectories must contain finite values.")
            return None
        if agent_mask.shape != trajectories.shape[:2]:
            raise ValueError("agent_mask must have shape [B, N].")
        if agent_mask.dtype != torch.bool:
            raise TypeError("agent_mask must use torch.bool.")
        if agent_mask.device != trajectories.device:
            raise ValueError("trajectories and agent_mask must share one device.")
        valid = trajectories[agent_mask]
        if valid.numel() and not torch.isfinite(valid).all():
            raise ValueError("Valid-agent trajectories must contain finite values.")
        return agent_mask

    @staticmethod
    def _zero_padding(trajectories: Tensor, agent_mask: Tensor | None) -> Tensor:
        if agent_mask is None:
            return trajectories
        return torch.where(
            agent_mask[..., None, None],
            trajectories,
            torch.zeros_like(trajectories),
        )

    def encode(
        self,
        physical_trajectories: Tensor,
        agent_mask: Tensor | None = None,
    ) -> Tensor:
        """Convert physical `[dx, dy, sin, cos]` paths to model space."""

        self._validate(physical_trajectories, agent_mask)
        encoded = self._zero_padding(physical_trajectories, agent_mask).clone()
        encoded[..., :2] /= self.position_scale
        return encoded

    def decode(
        self,
        model_trajectories: Tensor,
        agent_mask: Tensor | None = None,
    ) -> Tensor:
        """Return physical paths and unit-normalize valid heading vectors."""

        self._validate(model_trajectories, agent_mask)
        decoded = self._zero_padding(model_trajectories, agent_mask).clone()
        decoded[..., :2] *= self.position_scale

        heading = decoded[..., 2:4]
        heading_norm = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
        normalized_heading = heading / heading_norm.clamp_min(self.heading_epsilon)
        fallback = torch.zeros_like(normalized_heading)
        fallback[..., 1] = 1.0
        decoded[..., 2:4] = torch.where(
            heading_norm > self.heading_epsilon,
            normalized_heading,
            fallback,
        )
        return self._zero_padding(decoded, agent_mask)

