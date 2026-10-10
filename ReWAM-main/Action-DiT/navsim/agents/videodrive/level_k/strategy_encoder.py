"""Shared encoding of Ego and surrounding-agent strategy trajectories."""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch import Tensor, nn

from .types import AgentStateBatch, StrategyTokenBatch, StrategyTrajectoryBatch


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def ego_action_to_strategy(action_trajectory: Tensor) -> Tensor:
    """Convert physical `[x, y, heading]` Ego poses to canonical strategy paths."""

    if action_trajectory.ndim < 2 or action_trajectory.shape[-1] != 3:
        raise ValueError("action_trajectory must have shape [..., T, 3].")
    if not torch.is_floating_point(action_trajectory):
        raise TypeError("action_trajectory must use a floating-point dtype.")
    if not torch.isfinite(action_trajectory).all():
        raise ValueError("action_trajectory must contain finite physical values.")

    heading = action_trajectory[..., 2:3]
    return torch.cat(
        (
            action_trajectory[..., :2],
            torch.sin(heading),
            torch.cos(heading),
        ),
        dim=-1,
    )


class StrategyTokenEncoder(nn.Module):
    """Encode one complete trajectory per agent into one shared-space token."""

    EGO_ROLE = 0
    OTHER_ROLE = 1

    def __init__(
        self,
        *,
        horizon_steps: int,
        embedding_dim: int,
        max_reasoning_level: int,
        position_scale: float,
        velocity_scale: float = 30.0,
        size_scale: float = 10.0,
    ) -> None:
        super().__init__()
        if isinstance(horizon_steps, bool) or not isinstance(
            horizon_steps, Integral
        ):
            raise TypeError("horizon_steps must use an integer value.")
        if isinstance(embedding_dim, bool) or not isinstance(
            embedding_dim, Integral
        ):
            raise TypeError("embedding_dim must use an integer value.")
        if isinstance(max_reasoning_level, bool) or not isinstance(
            max_reasoning_level, Integral
        ):
            raise TypeError("max_reasoning_level must use an integer value.")
        for name, value in (
            ("position_scale", position_scale),
            ("velocity_scale", velocity_scale),
            ("size_scale", size_scale),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must use a real numeric value.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")

        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if max_reasoning_level < 0:
            raise ValueError("max_reasoning_level must be non-negative.")
        if position_scale <= 0:
            raise ValueError("position_scale must be positive.")
        if velocity_scale <= 0:
            raise ValueError("velocity_scale must be positive.")
        if size_scale <= 0:
            raise ValueError("size_scale must be positive.")

        self.horizon_steps = int(horizon_steps)
        self.embedding_dim = int(embedding_dim)
        self.max_reasoning_level = int(max_reasoning_level)
        self.position_scale = float(position_scale)
        self.velocity_scale = float(velocity_scale)
        self.size_scale = float(size_scale)

        self.trajectory_encoder = nn.Sequential(
            nn.Linear(horizon_steps * 4, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(8, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.role_embedding = nn.Embedding(2, embedding_dim)
        self.level_embedding = nn.Embedding(
            max_reasoning_level + 1,
            embedding_dim,
        )
        self.final_norm = nn.LayerNorm(embedding_dim)

        embedding_std = embedding_dim**-0.5
        nn.init.normal_(self.role_embedding.weight, mean=0.0, std=embedding_std)
        nn.init.normal_(self.level_embedding.weight, mean=0.0, std=embedding_std)

    def _level_ids(
        self,
        reasoning_level: int | Tensor,
        *,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
    ) -> Tensor:
        if isinstance(reasoning_level, Integral) and not isinstance(
            reasoning_level, bool
        ):
            levels = torch.full(
                (batch_size,),
                int(reasoning_level),
                device=device,
                dtype=torch.long,
            )
        elif isinstance(reasoning_level, Tensor):
            if reasoning_level.dtype not in _INTEGER_DTYPES:
                raise TypeError("reasoning_level tensor must use an integer dtype.")
            if reasoning_level.ndim == 0:
                levels = reasoning_level.to(device=device, dtype=torch.long).expand(
                    batch_size
                )
            elif reasoning_level.shape == (batch_size,):
                levels = reasoning_level.to(device=device, dtype=torch.long)
            else:
                raise ValueError(
                    "reasoning_level tensor must be scalar or have shape [B]."
                )
        else:
            raise TypeError("reasoning_level must be an int or integer tensor.")

        if torch.any(levels < 0) or torch.any(levels > self.max_reasoning_level):
            raise ValueError(
                "reasoning_level must satisfy "
                "0 <= level <= max_reasoning_level."
            )
        return levels[:, None].expand(batch_size, num_tokens)

    def forward(
        self,
        *,
        strategies: StrategyTrajectoryBatch,
        ego_velocity: Tensor,
        agent_states: AgentStateBatch,
        reasoning_level: int | Tensor,
    ) -> StrategyTokenBatch:
        """Return `[Ego, Other-0, ..., Other-N]` condition tokens."""

        strategies.validate(horizon_steps=self.horizon_steps)
        agent_states.validate()

        batch_size, num_agents, _ = agent_states.current_states.shape
        if strategies.other_trajectories.shape[:2] != (
            batch_size,
            num_agents,
        ):
            raise ValueError("strategies and agent_states must share B and N.")
        if not torch.equal(strategies.agent_mask, agent_states.agent_mask):
            raise ValueError("Strategy and agent-state masks must be identical.")
        if ego_velocity.shape != (batch_size, 2):
            raise ValueError("ego_velocity must have shape [B, 2].")
        if not torch.is_floating_point(ego_velocity):
            raise TypeError("ego_velocity must use a floating-point dtype.")

        reference = strategies.ego_trajectory
        tensors = (ego_velocity, agent_states.current_states)
        if any(tensor.device != reference.device for tensor in tensors):
            raise ValueError("Strategy encoder inputs must share one device.")
        if any(tensor.dtype != reference.dtype for tensor in tensors):
            raise TypeError("Floating-point strategy encoder inputs must share dtype.")
        if not torch.isfinite(ego_velocity).all():
            raise ValueError("ego_velocity must contain finite values.")

        agent_mask = agent_states.agent_mask
        valid_future_mask = strategies.valid_future_mask
        safe_other_trajectories = torch.where(
            valid_future_mask[..., None],
            strategies.other_trajectories,
            torch.zeros_like(strategies.other_trajectories),
        )
        safe_other_states = torch.where(
            agent_mask[..., None],
            agent_states.current_states,
            torch.zeros_like(agent_states.current_states),
        )

        trajectories = torch.cat(
            (strategies.ego_trajectory[:, None], safe_other_trajectories),
            dim=1,
        )
        ego_state = torch.zeros(
            (batch_size, 1, 8),
            device=reference.device,
            dtype=reference.dtype,
        )
        ego_state[..., 3] = 1.0
        ego_state[..., 4:6] = ego_velocity[:, None]
        states = torch.cat((ego_state, safe_other_states), dim=1)

        normalized_trajectories = trajectories.clone()
        normalized_trajectories[..., :2] /= self.position_scale
        normalized_states = states.clone()
        normalized_states[..., :2] /= self.position_scale
        normalized_states[..., 4:6] /= self.velocity_scale
        normalized_states[..., 6:8] /= self.size_scale

        num_tokens = num_agents + 1
        trajectory_tokens = self.trajectory_encoder(
            normalized_trajectories.reshape(batch_size, num_tokens, -1)
        )
        state_tokens = self.state_encoder(normalized_states)

        role_ids = torch.full(
            (batch_size, num_tokens),
            self.OTHER_ROLE,
            device=reference.device,
            dtype=torch.long,
        )
        role_ids[:, 0] = self.EGO_ROLE
        level_ids = self._level_ids(
            reasoning_level,
            batch_size=batch_size,
            num_tokens=num_tokens,
            device=reference.device,
        )

        tokens = self.final_norm(
            trajectory_tokens
            + state_tokens
            + self.role_embedding(role_ids)
            + self.level_embedding(level_ids)
        )
        strategy_agent_mask = valid_future_mask.any(dim=-1)
        token_mask = torch.cat(
            (
                torch.ones(
                    (batch_size, 1),
                    device=reference.device,
                    dtype=torch.bool,
                ),
                strategy_agent_mask,
            ),
            dim=1,
        )
        tokens = tokens * token_mask[..., None].to(dtype=tokens.dtype)

        output = StrategyTokenBatch(tokens=tokens, token_mask=token_mask)
        output.validate()
        return output
