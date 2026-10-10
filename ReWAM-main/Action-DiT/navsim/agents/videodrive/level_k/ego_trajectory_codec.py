"""Conversions between DriveLaW Action space and canonical Ego strategies."""

from __future__ import annotations

import torch
from torch import Tensor

from .strategy_encoder import ego_action_to_strategy


class EgoTrajectoryCodec:
    """Convert normalized Action trajectories to physical Ego-frame poses."""

    def __init__(
        self,
        *,
        x_offset: float = -1.57,
        x_span: float = 66.74,
        y_offset: float = -19.68,
        y_span: float = 42.0,
        heading_offset: float = -1.67,
        heading_span: float = 3.53,
    ) -> None:
        for name, value in (
            ("x_span", x_span),
            ("y_span", y_span),
            ("heading_span", heading_span),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        self.x_offset = x_offset
        self.x_span = x_span
        self.y_offset = y_offset
        self.y_span = y_span
        self.heading_offset = heading_offset
        self.heading_span = heading_span

    @staticmethod
    def _validate_action(trajectory: Tensor, *, name: str) -> None:
        if trajectory.ndim < 2 or trajectory.shape[-1] != 3:
            raise ValueError(f"{name} must have shape [..., T, 3].")
        if not torch.is_floating_point(trajectory):
            raise TypeError(f"{name} must use a floating-point dtype.")
        if not torch.isfinite(trajectory).all():
            raise ValueError(f"{name} must contain finite values.")

    @staticmethod
    def _validate_strategy(strategy: Tensor) -> None:
        if strategy.ndim < 2 or strategy.shape[-1] != 4:
            raise ValueError("strategy must have shape [..., T, 4].")
        if not torch.is_floating_point(strategy):
            raise TypeError("strategy must use a floating-point dtype.")
        if not torch.isfinite(strategy).all():
            raise ValueError("strategy must contain finite values.")

    def encode(self, physical_trajectory: Tensor) -> Tensor:
        """Normalize physical `[x, y, heading]` poses to Action model space."""

        self._validate_action(physical_trajectory, name="physical_trajectory")
        x = (
            2.0
            * (physical_trajectory[..., 0:1] - self.x_offset)
            / self.x_span
            - 1.0
        )
        y = (
            2.0
            * (physical_trajectory[..., 1:2] - self.y_offset)
            / self.y_span
            - 1.0
        )
        heading = (
            2.0
            * (physical_trajectory[..., 2:3] - self.heading_offset)
            / self.heading_span
            - 1.0
        )
        return torch.cat((x, y, heading), dim=-1)

    def decode(self, model_trajectory: Tensor) -> Tensor:
        """Denormalize Action model output to physical Ego-frame poses."""

        self._validate_action(model_trajectory, name="model_trajectory")
        x = (model_trajectory[..., 0:1] + 1.0) * 0.5 * self.x_span
        x = x + self.x_offset
        y = (model_trajectory[..., 1:2] + 1.0) * 0.5 * self.y_span
        y = y + self.y_offset
        heading = (
            (model_trajectory[..., 2:3] + 1.0)
            * 0.5
            * self.heading_span
        )
        heading = heading + self.heading_offset
        return torch.cat((x, y, heading), dim=-1)

    def model_to_strategy(self, model_trajectory: Tensor) -> Tensor:
        """Decode an Action trajectory and express heading as sine/cosine."""

        return ego_action_to_strategy(self.decode(model_trajectory))

    def strategy_to_physical(self, strategy: Tensor) -> Tensor:
        """Convert canonical `[x, y, sin, cos]` to physical poses."""

        self._validate_strategy(strategy)
        heading = torch.atan2(strategy[..., 2:3], strategy[..., 3:4])
        return torch.cat((strategy[..., :2], heading), dim=-1)

    def strategy_to_model(self, strategy: Tensor) -> Tensor:
        """Convert a canonical strategy to normalized Action model space."""

        return self.encode(self.strategy_to_physical(strategy))
