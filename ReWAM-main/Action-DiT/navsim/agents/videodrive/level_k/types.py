"""Tensor contracts shared by Level-k data, models, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


@dataclass
class AgentStateBatch:
    """Padded current surrounding-agent states without future supervision."""

    current_states: Tensor
    agent_types: Tensor
    agent_mask: Tensor

    @classmethod
    def from_target_dict(
        cls,
        values: Mapping[str, Tensor],
        *,
        add_batch_dim: bool = False,
    ) -> "AgentStateBatch":
        """Create current-only inputs from a feature or target dictionary."""

        tensors = {
            "current_states": values["agent_current_states"],
            "agent_types": values["agent_types"],
            "agent_mask": values["agent_mask"],
        }
        if add_batch_dim:
            tensors = {
                name: tensor.unsqueeze(0) for name, tensor in tensors.items()
            }
        return cls(**tensors)

    def validate(self, *, max_agents: int | None = None) -> None:
        """Validate ranks, dtypes, devices, and valid-agent state values."""

        if self.current_states.ndim != 3 or self.current_states.shape[-1] != 8:
            raise ValueError("current_states must have shape [B, N, 8].")

        batch_size, num_agents, _ = self.current_states.shape
        if self.agent_types.shape != (batch_size, num_agents):
            raise ValueError("agent_types must have shape [B, N].")
        if self.agent_mask.shape != (batch_size, num_agents):
            raise ValueError("agent_mask must have shape [B, N].")
        if max_agents is not None and num_agents != max_agents:
            raise ValueError(
                f"Expected N={max_agents} agents, but received N={num_agents}."
            )

        if not torch.is_floating_point(self.current_states):
            raise TypeError("current_states must use a floating-point dtype.")
        if self.agent_types.dtype not in _INTEGER_DTYPES:
            raise TypeError("agent_types must use an integer dtype.")
        if self.agent_mask.dtype != torch.bool:
            raise TypeError("agent_mask must use torch.bool.")
        if self.agent_types.device != self.current_states.device:
            raise ValueError("agent_types and current_states must share one device.")
        if self.agent_mask.device != self.current_states.device:
            raise ValueError("agent_mask and current_states must share one device.")

        valid_states = self.current_states[self.agent_mask]
        if valid_states.numel() == 0:
            return
        if not torch.isfinite(valid_states).all():
            raise ValueError("Valid-agent current_states must contain finite values.")

        heading_norm = torch.linalg.vector_norm(valid_states[:, 2:4], dim=-1)
        if torch.any(heading_norm <= 1e-6):
            raise ValueError(
                "Valid-agent sin/cos headings must have non-zero magnitude."
            )


@dataclass
class AgentTrajectoryBatch:
    """Padded surrounding-agent states and future trajectory supervision."""

    current_states: Tensor
    agent_types: Tensor
    agent_mask: Tensor
    future_trajectories: Tensor
    future_mask: Tensor

    def current_inputs(self) -> AgentStateBatch:
        """Return the current-only view used by inference-time policies."""

        return AgentStateBatch(
            current_states=self.current_states,
            agent_types=self.agent_types,
            agent_mask=self.agent_mask,
        )

    @classmethod
    def from_target_dict(
        cls,
        values: Mapping[str, Tensor],
        *,
        add_batch_dim: bool = False,
    ) -> "AgentTrajectoryBatch":
        """Create the contract from a target-builder dictionary."""

        tensors = {
            "current_states": values["agent_current_states"],
            "agent_types": values["agent_types"],
            "agent_mask": values["agent_mask"],
            "future_trajectories": values["agent_future_trajectories"],
            "future_mask": values["agent_future_mask"],
        }
        if add_batch_dim:
            tensors = {
                name: tensor.unsqueeze(0) for name, tensor in tensors.items()
            }
        return cls(**tensors)

    def validate(
        self,
        *,
        max_agents: int | None = None,
        horizon_steps: int | None = None,
    ) -> None:
        """Validate ranks, shared dimensions, dtypes, and devices."""

        self.current_inputs().validate(max_agents=max_agents)

        batch_size, num_agents, _ = self.current_states.shape

        if self.future_trajectories.ndim != 4 or self.future_trajectories.shape[-1] != 4:
            raise ValueError("future_trajectories must have shape [B, N, T, 4].")
        if self.future_trajectories.shape[:2] != (batch_size, num_agents):
            raise ValueError(
                "future_trajectories must share B and N with current_states."
            )

        num_steps = self.future_trajectories.shape[2]
        if self.future_mask.shape != (batch_size, num_agents, num_steps):
            raise ValueError("future_mask must have shape [B, N, T].")

        if horizon_steps is not None and num_steps != horizon_steps:
            raise ValueError(
                f"Expected T={horizon_steps} steps, but received T={num_steps}."
            )

        if not torch.is_floating_point(self.future_trajectories):
            raise TypeError("future_trajectories must use a floating-point dtype.")
        if self.future_mask.dtype != torch.bool:
            raise TypeError("future_mask must use torch.bool.")
        if self.future_trajectories.dtype != self.current_states.dtype:
            raise TypeError(
                "current_states and future_trajectories must share one dtype."
            )

        tensors = (
            self.agent_types,
            self.agent_mask,
            self.future_trajectories,
            self.future_mask,
        )
        if any(tensor.device != self.current_states.device for tensor in tensors):
            raise ValueError("All AgentTrajectoryBatch tensors must share one device.")
        if torch.any(self.future_mask & ~self.agent_mask.unsqueeze(-1)):
            raise ValueError(
                "Padded agents cannot have valid future trajectory steps."
            )

        valid_future = self.future_trajectories[self.future_mask]
        if valid_future.numel() == 0:
            return
        if not torch.isfinite(valid_future).all():
            raise ValueError(
                "Valid-agent future_trajectories must contain finite values."
            )
        heading_norm = torch.linalg.vector_norm(valid_future[:, 2:4], dim=-1)
        if torch.any(heading_norm <= 1e-6):
            raise ValueError(
                "Valid future sin/cos headings must have non-zero magnitude."
            )


@dataclass
class LevelZeroReferenceBatch:
    """Causal map references used to construct deterministic Level-0 paths.

    The Ego path is relative to the current Ego anchor.  Each surrounding-
    agent path is relative to that agent's current center, while every heading
    remains expressed in the current Ego frame.  A false reference mask means
    that the policy must use its constant-heading fallback for the complete
    trajectory; map and fallback paths are never spliced within one horizon.
    """

    ego_trajectory: Tensor
    ego_reference_mask: Tensor
    other_trajectories: Tensor
    other_reference_mask: Tensor

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Tensor],
        *,
        add_batch_dim: bool = False,
    ) -> "LevelZeroReferenceBatch":
        tensors = {
            "ego_trajectory": values["level_zero_ego_reference"],
            "ego_reference_mask": values[
                "level_zero_ego_reference_mask"
            ],
            "other_trajectories": values[
                "level_zero_other_references"
            ],
            "other_reference_mask": values[
                "level_zero_other_reference_mask"
            ],
        }
        if add_batch_dim:
            tensors = {
                name: tensor.unsqueeze(0) for name, tensor in tensors.items()
            }
        return cls(**tensors)

    def validate(
        self,
        *,
        agent_mask: Tensor | None = None,
        horizon_steps: int | None = None,
    ) -> None:
        if self.ego_trajectory.ndim != 3 or self.ego_trajectory.shape[-1] != 4:
            raise ValueError("ego_trajectory must have shape [B, T, 4].")
        if (
            self.other_trajectories.ndim != 4
            or self.other_trajectories.shape[-1] != 4
        ):
            raise ValueError("other_trajectories must have shape [B, N, T, 4].")

        batch_size, num_steps, _ = self.ego_trajectory.shape
        other_batch, num_agents, other_steps, _ = self.other_trajectories.shape
        if other_batch != batch_size or other_steps != num_steps:
            raise ValueError(
                "Ego and Other Level-0 references must share B and T."
            )
        if self.ego_reference_mask.shape != (batch_size,):
            raise ValueError("ego_reference_mask must have shape [B].")
        if self.other_reference_mask.shape != (batch_size, num_agents):
            raise ValueError("other_reference_mask must have shape [B, N].")
        if horizon_steps is not None and num_steps != horizon_steps:
            raise ValueError(
                f"Expected T={horizon_steps} steps, but received T={num_steps}."
            )

        if not torch.is_floating_point(self.ego_trajectory):
            raise TypeError("ego_trajectory must use a floating-point dtype.")
        if not torch.is_floating_point(self.other_trajectories):
            raise TypeError("other_trajectories must use a floating-point dtype.")
        if self.ego_trajectory.dtype != self.other_trajectories.dtype:
            raise TypeError("Level-0 reference trajectories must share one dtype.")
        if self.ego_reference_mask.dtype != torch.bool:
            raise TypeError("ego_reference_mask must use torch.bool.")
        if self.other_reference_mask.dtype != torch.bool:
            raise TypeError("other_reference_mask must use torch.bool.")

        tensors = (
            self.ego_reference_mask,
            self.other_trajectories,
            self.other_reference_mask,
        )
        if any(tensor.device != self.ego_trajectory.device for tensor in tensors):
            raise ValueError("All Level-0 reference tensors must share one device.")

        if agent_mask is not None:
            if agent_mask.shape != (batch_size, num_agents):
                raise ValueError("agent_mask must have shape [B, N].")
            if agent_mask.dtype != torch.bool:
                raise TypeError("agent_mask must use torch.bool.")
            if agent_mask.device != self.ego_trajectory.device:
                raise ValueError("agent_mask and references must share one device.")
            if torch.any(self.other_reference_mask & ~agent_mask):
                raise ValueError("Padded agents cannot have valid map references.")

        valid_ego = self.ego_trajectory[self.ego_reference_mask]
        valid_other = self.other_trajectories[self.other_reference_mask]
        for name, valid in (("Ego", valid_ego), ("Other", valid_other)):
            if valid.numel() == 0:
                continue
            if not torch.isfinite(valid).all():
                raise ValueError(f"Valid {name} map references must be finite.")
            heading_norm = torch.linalg.vector_norm(valid[..., 2:4], dim=-1)
            if torch.any(heading_norm <= 1e-6):
                raise ValueError(
                    f"Valid {name} map-reference headings must be non-zero."
                )

        invalid_other = self.other_trajectories[
            ~self.other_reference_mask
        ]
        if invalid_other.numel() and torch.count_nonzero(invalid_other):
            raise ValueError("Invalid Other map-reference slots must be zero.")


@dataclass
class StrategyTrajectoryBatch:
    """Canonical Ego and other-agent strategies in the current Ego frame."""

    ego_trajectory: Tensor
    other_trajectories: Tensor
    agent_mask: Tensor
    future_mask: Tensor | None = None

    @property
    def valid_future_mask(self) -> Tensor:
        """Return the effective per-agent/per-time strategy validity mask."""

        batch_size, num_agents, num_steps, _ = self.other_trajectories.shape
        if self.future_mask is None:
            future_mask = torch.ones(
                (batch_size, num_agents, num_steps),
                device=self.other_trajectories.device,
                dtype=torch.bool,
            )
        else:
            future_mask = self.future_mask
        return self.agent_mask[..., None] & future_mask

    def validate(self, *, horizon_steps: int | None = None) -> None:
        """Validate canonical `[dx, dy, sin(heading), cos(heading)]` paths."""

        if self.ego_trajectory.ndim != 3 or self.ego_trajectory.shape[-1] != 4:
            raise ValueError("ego_trajectory must have shape [B, T, 4].")
        if (
            self.other_trajectories.ndim != 4
            or self.other_trajectories.shape[-1] != 4
        ):
            raise ValueError("other_trajectories must have shape [B, N, T, 4].")

        batch_size, num_steps, _ = self.ego_trajectory.shape
        other_batch, num_agents, other_steps, _ = self.other_trajectories.shape
        if other_batch != batch_size or other_steps != num_steps:
            raise ValueError(
                "Ego and other-agent trajectories must share B and T."
            )
        if self.agent_mask.shape != (batch_size, num_agents):
            raise ValueError("agent_mask must have shape [B, N].")
        if horizon_steps is not None and num_steps != horizon_steps:
            raise ValueError(
                f"Expected T={horizon_steps} steps, but received T={num_steps}."
            )

        if not torch.is_floating_point(self.ego_trajectory):
            raise TypeError("ego_trajectory must use a floating-point dtype.")
        if not torch.is_floating_point(self.other_trajectories):
            raise TypeError("other_trajectories must use a floating-point dtype.")
        if self.agent_mask.dtype != torch.bool:
            raise TypeError("agent_mask must use torch.bool.")
        if self.other_trajectories.device != self.ego_trajectory.device:
            raise ValueError("Strategy trajectories must share one device.")
        if self.agent_mask.device != self.ego_trajectory.device:
            raise ValueError("Strategy trajectories and mask must share one device.")
        if self.other_trajectories.dtype != self.ego_trajectory.dtype:
            raise TypeError("Strategy trajectories must share one dtype.")

        if self.future_mask is not None:
            if self.future_mask.shape != (batch_size, num_agents, num_steps):
                raise ValueError("future_mask must have shape [B, N, T].")
            if self.future_mask.dtype != torch.bool:
                raise TypeError("future_mask must use torch.bool.")
            if self.future_mask.device != self.ego_trajectory.device:
                raise ValueError(
                    "Strategy trajectories and future_mask must share one device."
                )
            if torch.any(self.future_mask & ~self.agent_mask.unsqueeze(-1)):
                raise ValueError(
                    "Padded agents cannot have valid strategy future steps."
                )

        if not torch.isfinite(self.ego_trajectory).all():
            raise ValueError("ego_trajectory must contain finite values.")
        valid_other = self.other_trajectories[self.valid_future_mask]
        if valid_other.numel() and not torch.isfinite(valid_other).all():
            raise ValueError(
                "Valid-agent other_trajectories must contain finite values."
            )


@dataclass
class StrategyTokenBatch:
    """One condition token per Ego/agent strategy, with Ego at index zero."""

    tokens: Tensor
    token_mask: Tensor

    @property
    def ego_token(self) -> Tensor:
        return self.tokens[:, 0]

    @property
    def other_tokens(self) -> Tensor:
        return self.tokens[:, 1:]

    @property
    def ego_mask(self) -> Tensor:
        return self.token_mask[:, 0]

    @property
    def other_mask(self) -> Tensor:
        return self.token_mask[:, 1:]

    def validate(self) -> None:
        """Validate token ordering, mask semantics, and padded-token zeros."""

        if self.tokens.ndim != 3:
            raise ValueError("tokens must have shape [B, 1+N, D].")
        batch_size, num_tokens, _ = self.tokens.shape
        if num_tokens < 1:
            raise ValueError("tokens must contain the Ego token at index zero.")
        if self.token_mask.shape != (batch_size, num_tokens):
            raise ValueError("token_mask must have shape [B, 1+N].")
        if not torch.is_floating_point(self.tokens):
            raise TypeError("tokens must use a floating-point dtype.")
        if self.token_mask.dtype != torch.bool:
            raise TypeError("token_mask must use torch.bool.")
        if self.token_mask.device != self.tokens.device:
            raise ValueError("tokens and token_mask must share one device.")
        if not torch.all(self.ego_mask):
            raise ValueError("The Ego token at index zero must always be valid.")

        valid_tokens = self.tokens[self.token_mask]
        if not torch.isfinite(valid_tokens).all():
            raise ValueError("Valid strategy tokens must contain finite values.")
        padded_tokens = self.tokens[~self.token_mask]
        if padded_tokens.numel() and torch.count_nonzero(padded_tokens):
            raise ValueError("Padded strategy tokens must be exactly zero.")


@dataclass
class LevelKOutput:
    """Trajectories retained at every reasoning level."""

    ego_levels: Sequence[Tensor]
    other_levels: Sequence[Tensor]
    reasoning_level: int

    def validate(self) -> None:
        """Validate the retained recursion depth and trajectory ranks."""

        if self.reasoning_level < 0:
            raise ValueError("reasoning_level must be non-negative.")

        required_levels = self.reasoning_level + 1
        if len(self.ego_levels) < required_levels:
            raise ValueError("ego_levels does not contain the requested level.")
        if len(self.other_levels) < required_levels:
            raise ValueError("other_levels does not contain the requested level.")

        reference_ego_shape = self.ego_levels[0].shape
        reference_other_shape = self.other_levels[0].shape
        reference_device = self.ego_levels[0].device
        reference_dtype = self.ego_levels[0].dtype

        for trajectory in self.ego_levels[:required_levels]:
            if trajectory.ndim != 3 or trajectory.shape[-1] != 4:
                raise ValueError(
                    "Each ego strategy must have shape [B, T, 4]."
                )
            if trajectory.shape != reference_ego_shape:
                raise ValueError("All Ego levels must share one shape.")
            if trajectory.device != reference_device:
                raise ValueError("All reasoning levels must share one device.")
            if trajectory.dtype != reference_dtype:
                raise TypeError("All reasoning levels must share one dtype.")
            if not torch.isfinite(trajectory).all():
                raise ValueError("Ego strategies must contain finite values.")

        for trajectories in self.other_levels[:required_levels]:
            if trajectories.ndim != 4 or trajectories.shape[-1] != 4:
                raise ValueError(
                    "Each Other strategy must have shape [B, N, T, 4]."
                )
            if trajectories.shape != reference_other_shape:
                raise ValueError("All Other levels must share one shape.")
            if trajectories.shape[0] != reference_ego_shape[0]:
                raise ValueError("Ego and Other levels must share B.")
            if trajectories.shape[2] != reference_ego_shape[1]:
                raise ValueError("Ego and Other levels must share T.")
            if trajectories.device != reference_device:
                raise ValueError("All reasoning levels must share one device.")
            if trajectories.dtype != reference_dtype:
                raise TypeError("All reasoning levels must share one dtype.")
            if not torch.isfinite(trajectories).all():
                raise ValueError("Other strategies must contain finite values.")
