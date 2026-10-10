"""Tests for shared strategy trajectory condition tokens."""

from __future__ import annotations

import math

import pytest
import torch

from navsim.agents.videodrive.level_k import (
    AgentStateBatch,
    LevelZeroPolicy,
    StrategyTokenEncoder,
    StrategyTrajectoryBatch,
    ego_action_to_strategy,
)


def _inputs(
    *,
    batch_size: int = 2,
    num_agents: int = 10,
    horizon_steps: int = 8,
    dtype: torch.dtype = torch.float32,
) -> tuple[StrategyTrajectoryBatch, torch.Tensor, AgentStateBatch]:
    states = torch.zeros(batch_size, num_agents, 8, dtype=dtype)
    states[..., 0] = torch.arange(num_agents, dtype=dtype) + 5.0
    states[..., 3] = 1.0
    states[..., 4] = 4.0
    states[..., 6] = 4.5
    states[..., 7] = 2.0
    agent_types = torch.zeros(batch_size, num_agents, dtype=torch.int64)
    agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
    agents = AgentStateBatch(states, agent_types, agent_mask)
    ego_velocity = torch.full((batch_size, 2), 0.0, dtype=dtype)
    ego_velocity[:, 0] = 6.0
    strategies = LevelZeroPolicy(
        horizon_steps=horizon_steps,
        interval_length=0.5,
    )(ego_velocity, agents)
    return strategies, ego_velocity, agents


def _encoder(
    *,
    horizon_steps: int = 8,
    embedding_dim: int = 32,
) -> StrategyTokenEncoder:
    return StrategyTokenEncoder(
        horizon_steps=horizon_steps,
        embedding_dim=embedding_dim,
        max_reasoning_level=3,
        position_scale=80.0,
        velocity_scale=30.0,
        size_scale=10.0,
    )


@pytest.mark.parametrize("num_agents", [8, 10])
def test_token_shapes_and_ego_order(num_agents: int) -> None:
    strategies, ego_velocity, agents = _inputs(num_agents=num_agents)
    output = _encoder()(
        strategies=strategies,
        ego_velocity=ego_velocity,
        agent_states=agents,
        reasoning_level=0,
    )

    assert output.tokens.shape == (2, num_agents + 1, 32)
    assert output.token_mask.shape == (2, num_agents + 1)
    assert output.ego_token.shape == (2, 32)
    assert output.other_tokens.shape == (2, num_agents, 32)
    assert torch.all(output.ego_mask)
    torch.testing.assert_close(output.other_mask, agents.agent_mask)


def test_padding_tokens_are_exactly_zero() -> None:
    strategies, ego_velocity, agents = _inputs()
    agents.agent_mask[:, -2:] = False
    strategies.agent_mask[:, -2:] = False
    output = _encoder()(
        strategies=strategies,
        ego_velocity=ego_velocity,
        agent_states=agents,
        reasoning_level=0,
    )

    assert torch.count_nonzero(output.other_tokens[:, -2:]) == 0
    assert not torch.any(output.other_mask[:, -2:])


def test_identical_other_agents_use_shared_weights() -> None:
    strategies, ego_velocity, agents = _inputs(batch_size=1)
    agents.current_states[:, 1] = agents.current_states[:, 0]
    strategies.other_trajectories[:, 1] = strategies.other_trajectories[:, 0]
    output = _encoder()(
        strategies=strategies,
        ego_velocity=ego_velocity,
        agent_states=agents,
        reasoning_level=0,
    )

    torch.testing.assert_close(output.other_tokens[:, 0], output.other_tokens[:, 1])


def test_other_agent_permutation_equivariance() -> None:
    strategies, ego_velocity, agents = _inputs(batch_size=1)
    encoder = _encoder().eval()
    original = encoder(
        strategies=strategies,
        ego_velocity=ego_velocity,
        agent_states=agents,
        reasoning_level=1,
    )

    permutation = torch.tensor([2, 0, 1, 3, 4, 5, 6, 7, 8, 9])
    permuted_strategies = StrategyTrajectoryBatch(
        ego_trajectory=strategies.ego_trajectory,
        other_trajectories=strategies.other_trajectories[:, permutation],
        agent_mask=strategies.agent_mask[:, permutation],
    )
    permuted_agents = AgentStateBatch(
        current_states=agents.current_states[:, permutation],
        agent_types=agents.agent_types[:, permutation],
        agent_mask=agents.agent_mask[:, permutation],
    )
    permuted = encoder(
        strategies=permuted_strategies,
        ego_velocity=ego_velocity,
        agent_states=permuted_agents,
        reasoning_level=1,
    )

    torch.testing.assert_close(permuted.ego_token, original.ego_token)
    torch.testing.assert_close(
        permuted.other_tokens,
        original.other_tokens[:, permutation],
    )


def test_temporal_order_changes_token() -> None:
    strategies, ego_velocity, agents = _inputs(batch_size=1)
    encoder = _encoder().eval()
    forward = encoder(
        strategies=strategies,
        ego_velocity=ego_velocity,
        agent_states=agents,
        reasoning_level=0,
    )
    reversed_strategies = StrategyTrajectoryBatch(
        ego_trajectory=strategies.ego_trajectory.flip(1),
        other_trajectories=strategies.other_trajectories.flip(2),
        agent_mask=strategies.agent_mask,
    )
    reversed_output = encoder(
        strategies=reversed_strategies,
        ego_velocity=ego_velocity,
        agent_states=agents,
        reasoning_level=0,
    )

    assert not torch.allclose(forward.tokens, reversed_output.tokens)


def test_role_embedding_distinguishes_ego_and_other() -> None:
    strategies, ego_velocity, agents = _inputs(batch_size=1)
    agents.current_states[:, 0] = 0.0
    agents.current_states[:, 0, 3] = 1.0
    agents.current_states[:, 0, 4:6] = ego_velocity
    strategies.other_trajectories[:, 0] = strategies.ego_trajectory
    encoder = _encoder().eval()
    output = encoder(
        strategies=strategies,
        ego_velocity=ego_velocity,
        agent_states=agents,
        reasoning_level=0,
    )

    assert not torch.allclose(output.ego_token, output.other_tokens[:, 0])


def test_level_embedding_changes_token_and_supports_per_batch_level() -> None:
    strategies, ego_velocity, agents = _inputs()
    encoder = _encoder().eval()
    level_zero = encoder(
        strategies=strategies,
        ego_velocity=ego_velocity,
        agent_states=agents,
        reasoning_level=0,
    )
    mixed = encoder(
        strategies=strategies,
        ego_velocity=ego_velocity,
        agent_states=agents,
        reasoning_level=torch.tensor([0, 2]),
    )

    torch.testing.assert_close(level_zero.tokens[0], mixed.tokens[0])
    assert not torch.allclose(level_zero.tokens[1], mixed.tokens[1])


def test_rejects_invalid_reasoning_level() -> None:
    strategies, ego_velocity, agents = _inputs()
    encoder = _encoder()

    with pytest.raises(ValueError, match="reasoning_level"):
        encoder(
            strategies=strategies,
            ego_velocity=ego_velocity,
            agent_states=agents,
            reasoning_level=4,
        )
    with pytest.raises(TypeError, match="integer"):
        encoder(
            strategies=strategies,
            ego_velocity=ego_velocity,
            agent_states=agents,
            reasoning_level=torch.tensor([0.0, 1.0]),
        )


def test_ego_action_to_strategy_handles_wrapped_angles() -> None:
    actions = torch.tensor(
        [[[1.0, 2.0, math.pi], [3.0, 4.0, -math.pi + 1e-5]]],
        dtype=torch.float64,
    )
    output = ego_action_to_strategy(actions)

    assert output.shape == (1, 2, 4)
    torch.testing.assert_close(output[..., :2], actions[..., :2])
    torch.testing.assert_close(output[..., 2], torch.sin(actions[..., 2]))
    torch.testing.assert_close(output[..., 3], torch.cos(actions[..., 2]))


def test_dtype_and_gradients() -> None:
    strategies, ego_velocity, agents = _inputs(
        batch_size=1,
        dtype=torch.float64,
    )
    strategies.ego_trajectory.requires_grad_()
    strategies.other_trajectories.requires_grad_()
    ego_velocity.requires_grad_()
    agents.current_states.requires_grad_()
    encoder = _encoder().to(dtype=torch.float64)

    output = encoder(
        strategies=strategies,
        ego_velocity=ego_velocity,
        agent_states=agents,
        reasoning_level=1,
    )
    output.tokens.square().sum().backward()

    assert output.tokens.dtype == torch.float64
    assert strategies.ego_trajectory.grad is not None
    assert strategies.other_trajectories.grad is not None
    assert ego_velocity.grad is not None
    assert agents.current_states.grad is not None
    assert encoder.role_embedding.weight.grad is not None
    assert encoder.level_embedding.weight.grad is not None


def test_masked_nan_is_sanitized_but_valid_nan_is_rejected() -> None:
    strategies, ego_velocity, agents = _inputs(batch_size=1)
    agents.agent_mask[:, -1] = False
    strategies.agent_mask[:, -1] = False
    agents.current_states[:, -1] = float("nan")
    strategies.other_trajectories[:, -1] = float("nan")
    encoder = _encoder()

    output = encoder(
        strategies=strategies,
        ego_velocity=ego_velocity,
        agent_states=agents,
        reasoning_level=0,
    )
    assert torch.count_nonzero(output.other_tokens[:, -1]) == 0

    agents.current_states[:, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        encoder(
            strategies=strategies,
            ego_velocity=ego_velocity,
            agent_states=agents,
            reasoning_level=0,
        )


def test_future_mask_sanitizes_gaps_and_hides_empty_strategy_slots() -> None:
    strategies, ego_velocity, agents = _inputs(batch_size=1, num_agents=2)
    strategies.other_trajectories[:, 0, 1] = float("nan")
    strategies.future_mask = torch.zeros(1, 2, 8, dtype=torch.bool)
    strategies.future_mask[:, 0, 0] = True
    encoder = _encoder().eval()

    output = encoder(
        strategies=strategies,
        ego_velocity=ego_velocity,
        agent_states=agents,
        reasoning_level=0,
    )

    assert output.other_mask.tolist() == [[True, False]]
    assert torch.isfinite(output.tokens).all()
    assert torch.count_nonzero(output.other_tokens[:, 1]) == 0


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"horizon_steps": True}, TypeError),
        ({"horizon_steps": 8.5}, TypeError),
        ({"embedding_dim": True}, TypeError),
        ({"max_reasoning_level": 3.0}, TypeError),
        ({"position_scale": True}, TypeError),
        ({"position_scale": float("nan")}, ValueError),
        ({"velocity_scale": float("inf")}, ValueError),
        ({"size_scale": float("nan")}, ValueError),
    ],
)
def test_encoder_rejects_invalid_constructor_values(
    kwargs: dict,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        StrategyTokenEncoder(
            horizon_steps=kwargs.pop("horizon_steps", 8),
            embedding_dim=kwargs.pop("embedding_dim", 32),
            max_reasoning_level=kwargs.pop("max_reasoning_level", 3),
            position_scale=kwargs.pop("position_scale", 80.0),
            velocity_scale=kwargs.pop("velocity_scale", 30.0),
            size_scale=kwargs.pop("size_scale", 10.0),
        )
