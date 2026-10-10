"""Deterministic Level-0 policies for Ego and surrounding vehicles."""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch import Tensor, nn

from .types import (
    AgentStateBatch,
    LevelZeroReferenceBatch,
    StrategyTrajectoryBatch,
)


class LevelZeroStrategies(StrategyTrajectoryBatch):
    """Canonical Level-0 trajectories in the current Ego anchor frame."""


class LevelZeroPolicy(nn.Module):
    """Constant-speed, constant-heading policy with no trainable parameters."""

    def __init__(
        self,
        *,
        horizon_steps: int = 8,
        interval_length: float = 0.5,
        heading_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if isinstance(horizon_steps, bool) or not isinstance(
            horizon_steps,
            Integral,
        ):
            raise TypeError("horizon_steps must use an integer value.")
        for name, value in (
            ("interval_length", interval_length),
            ("heading_epsilon", heading_epsilon),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must use a real numeric value.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive.")
        if interval_length <= 0:
            raise ValueError("interval_length must be positive.")
        if heading_epsilon <= 0:
            raise ValueError("heading_epsilon must be positive.")

        self.horizon_steps = int(horizon_steps)
        self.interval_length = float(interval_length)
        self.heading_epsilon = float(heading_epsilon)

    def forward(
        self,
        ego_velocity: Tensor,
        agents: AgentStateBatch,
        references: LevelZeroReferenceBatch | None = None,
    ) -> LevelZeroStrategies:
        """Generate canonical `[dx, dy, sin(heading), cos(heading)]` trajectories."""

        agents.validate()
        if ego_velocity.ndim != 2 or ego_velocity.shape[-1] != 2:
            raise ValueError("ego_velocity must have shape [B, 2].")
        if ego_velocity.shape[0] != agents.current_states.shape[0]:
            raise ValueError("ego_velocity and agents must share batch size B.")
        if not torch.is_floating_point(ego_velocity):
            raise TypeError("ego_velocity must use a floating-point dtype.")
        if ego_velocity.device != agents.current_states.device:
            raise ValueError("ego_velocity and agent states must share one device.")
        if ego_velocity.dtype != agents.current_states.dtype:
            raise TypeError("ego_velocity and agent states must share one dtype.")
        if not torch.isfinite(ego_velocity).all():
            raise ValueError("ego_velocity must contain finite values.")

        agent_state_mask = agents.agent_mask[..., None]
        current_states = torch.where(
            agent_state_mask,
            agents.current_states,
            torch.zeros_like(agents.current_states),
        )
        batch_size, num_agents, _ = current_states.shape
        times = (
            torch.arange(
                1,
                self.horizon_steps + 1,
                device=current_states.device,
                dtype=current_states.dtype,
            )
            * self.interval_length
        )

        ego_speed = ego_velocity[:, 0].clamp_min(0.0)
        ego_trajectory = torch.zeros(
            (batch_size, self.horizon_steps, 4),
            device=current_states.device,
            dtype=current_states.dtype,
        )
        ego_trajectory[..., 0] = ego_speed[:, None] * times[None, :]
        ego_trajectory[..., 3] = 1.0

        heading_sin_cos = current_states[..., 2:4]
        heading_norm = torch.linalg.vector_norm(heading_sin_cos, dim=-1)
        safe_heading_norm = heading_norm.clamp_min(self.heading_epsilon)
        heading_sin = heading_sin_cos[..., 0] / safe_heading_norm
        heading_cos = heading_sin_cos[..., 1] / safe_heading_norm

        velocity = current_states[..., 4:6]
        longitudinal_speed = (
            velocity[..., 0] * heading_cos + velocity[..., 1] * heading_sin
        ).clamp_min(0.0)
        travel_distance = longitudinal_speed[..., None] * times[None, None, :]

        other_trajectories = torch.stack(
            (
                travel_distance * heading_cos[..., None],
                travel_distance * heading_sin[..., None],
                heading_sin[..., None].expand(-1, -1, self.horizon_steps),
                heading_cos[..., None].expand(-1, -1, self.horizon_steps),
            ),
            dim=-1,
        ).to(dtype=ego_trajectory.dtype)
        trajectory_mask = agents.agent_mask[..., None, None]
        other_trajectories = torch.where(
            trajectory_mask,
            other_trajectories,
            torch.zeros_like(other_trajectories),
        )

        if references is not None:
            references.validate(
                agent_mask=agents.agent_mask,
                horizon_steps=self.horizon_steps,
            )
            if references.ego_trajectory.shape[0] != batch_size:
                raise ValueError(
                    "Level-0 references and agent states must share batch size B."
                )
            if references.ego_trajectory.device != current_states.device:
                raise ValueError(
                    "Level-0 references and agent states must share one device."
                )
            if references.ego_trajectory.dtype != current_states.dtype:
                raise TypeError(
                    "Level-0 references and agent states must share one dtype."
                )
            ego_trajectory = torch.where(
                references.ego_reference_mask[:, None, None],
                references.ego_trajectory,
                ego_trajectory,
            )
            other_trajectories = torch.where(
                references.other_reference_mask[..., None, None],
                references.other_trajectories,
                other_trajectories,
            )

        strategies = LevelZeroStrategies(
            ego_trajectory=ego_trajectory,
            other_trajectories=other_trajectories,
            agent_mask=agents.agent_mask,
        )
        strategies.validate()
        return strategies
