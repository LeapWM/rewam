"""State management for synchronous Level-k best-response recursion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from .config import LevelKConfig
from .level_zero import LevelZeroPolicy
from .types import (
    AgentStateBatch,
    LevelZeroReferenceBatch,
    LevelKOutput,
    StrategyTrajectoryBatch,
    StrategyTokenBatch,
)


@dataclass
class LevelKReasoningState:
    """Canonical trajectories retained after every completed reasoning level."""

    ego_levels: list[Tensor]
    other_levels: list[Tensor]
    agent_mask: Tensor

    @property
    def current_level(self) -> int:
        return len(self.ego_levels) - 1

    @property
    def current_strategies(self) -> StrategyTrajectoryBatch:
        strategies = StrategyTrajectoryBatch(
            ego_trajectory=self.ego_levels[-1],
            other_trajectories=self.other_levels[-1],
            agent_mask=self.agent_mask,
        )
        strategies.validate()
        return strategies

    @classmethod
    def from_level_zero(
        cls,
        strategies: StrategyTrajectoryBatch,
    ) -> "LevelKReasoningState":
        strategies.validate()
        return cls(
            ego_levels=[strategies.ego_trajectory],
            other_levels=[strategies.other_trajectories],
            agent_mask=strategies.agent_mask,
        )

    def append(
        self,
        *,
        ego_strategy: Tensor,
        other_strategies: Tensor,
    ) -> None:
        candidate = StrategyTrajectoryBatch(
            ego_trajectory=ego_strategy,
            other_trajectories=other_strategies,
            agent_mask=self.agent_mask,
        )
        candidate.validate(
            horizon_steps=self.ego_levels[0].shape[1],
        )
        if ego_strategy.shape != self.ego_levels[0].shape:
            raise ValueError("Every Ego level must use the Level-0 shape.")
        if other_strategies.shape != self.other_levels[0].shape:
            raise ValueError("Every Other level must use the Level-0 shape.")
        if ego_strategy.device != self.ego_levels[0].device:
            raise ValueError("Every reasoning level must share one device.")
        if ego_strategy.dtype != self.ego_levels[0].dtype:
            raise TypeError("Every reasoning level must share one dtype.")
        if other_strategies.device != self.other_levels[0].device:
            raise ValueError("Every reasoning level must share one device.")
        if other_strategies.dtype != self.other_levels[0].dtype:
            raise TypeError("Every reasoning level must share one dtype.")

        padded = other_strategies[~self.agent_mask]
        if padded.numel() and torch.count_nonzero(padded):
            raise ValueError("Padded Other trajectories must be exactly zero.")
        self.ego_levels.append(ego_strategy)
        self.other_levels.append(other_strategies)

    def output(self, reasoning_level: int | None = None) -> LevelKOutput:
        if reasoning_level is None:
            reasoning_level = self.current_level
        if not 0 <= reasoning_level <= self.current_level:
            raise ValueError("Requested reasoning level has not been completed.")
        output = LevelKOutput(
            ego_levels=tuple(self.ego_levels[: reasoning_level + 1]),
            other_levels=tuple(self.other_levels[: reasoning_level + 1]),
            reasoning_level=reasoning_level,
        )
        output.validate()
        return output


class LevelKReasoner:
    """Initialize and validate Level-k recursion without owning neural experts."""

    def __init__(self, config: LevelKConfig) -> None:
        self.config = config
        self.level_zero_policy = LevelZeroPolicy(
            horizon_steps=config.horizon_steps,
            interval_length=config.interval_length,
        )

    def initialize(
        self,
        *,
        ego_velocity: Tensor,
        agents: AgentStateBatch,
        references: LevelZeroReferenceBatch | None = None,
    ) -> LevelKReasoningState:
        agents.validate(max_agents=self.config.max_agents)
        strategies = self.level_zero_policy(ego_velocity, agents, references)
        return LevelKReasoningState.from_level_zero(strategies)

    def validate_complete(self, state: LevelKReasoningState) -> None:
        if state.current_level != self.config.reasoning_level:
            raise ValueError(
                "Reasoning state ended at level "
                f"{state.current_level}, expected {self.config.reasoning_level}."
            )
        state.output(self.config.reasoning_level)

    def run(
        self,
        *,
        state: LevelKReasoningState,
        encode_strategy: Callable[
            [StrategyTrajectoryBatch, int],
            StrategyTokenBatch,
        ],
        predict_best_responses: Callable[
            [StrategyTokenBatch, int],
            tuple[Tensor, Tensor],
        ],
    ) -> LevelKOutput:
        """Run synchronous recursion using fixed previous-level conditions."""

        if state.current_level != 0:
            raise ValueError("Level-k recursion must start from Level-0.")
        for target_level in range(1, self.config.reasoning_level + 1):
            source_level = target_level - 1
            if state.current_level != source_level:
                raise RuntimeError("Reasoning state advanced out of order.")
            fixed_previous_strategies = state.current_strategies
            strategy_tokens = encode_strategy(
                fixed_previous_strategies,
                source_level,
            )
            strategy_tokens.validate()
            ego_strategy, other_strategies = predict_best_responses(
                strategy_tokens,
                target_level,
            )
            state.append(
                ego_strategy=ego_strategy,
                other_strategies=other_strategies,
            )
        self.validate_complete(state)
        return state.output(self.config.reasoning_level)
