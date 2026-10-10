import math

import pytest
import torch

from navsim.agents.videodrive.level_k import (
    AgentStateBatch,
    EgoTrajectoryCodec,
    LevelKConfig,
    LevelKReasoner,
    StrategyTokenBatch,
)


def _agents(*, batch_size: int = 2, num_agents: int = 3):
    states = torch.zeros(batch_size, num_agents, 8)
    states[..., 1] = torch.arange(num_agents, dtype=torch.float32)
    states[..., 3] = 1.0
    states[..., 4] = 2.0
    states[..., 6:8] = torch.tensor([4.5, 2.0])
    mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
    mask[1, 1:] = False
    states = states * mask[..., None]
    types = torch.where(
        mask,
        torch.zeros_like(mask, dtype=torch.long),
        torch.full_like(mask, -1, dtype=torch.long),
    )
    return AgentStateBatch(
        current_states=states,
        agent_types=types,
        agent_mask=mask,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ego_codec_round_trip_matches_legacy_formula(dtype):
    codec = EgoTrajectoryCodec()
    physical = torch.tensor(
        [
            [[0.0, -2.0, -math.pi / 2], [20.0, 4.0, math.pi / 3]],
            [[-1.57, -19.68, -1.67], [65.17, 22.32, 1.86]],
        ],
        dtype=dtype,
    )
    model = codec.encode(physical)
    legacy = torch.cat(
        (
            2 * (physical[..., 0:1] + 1.57) / 66.74 - 1,
            2 * (physical[..., 1:2] + 19.68) / 42.0 - 1,
            2 * (physical[..., 2:3] + 1.67) / 3.53 - 1,
        ),
        dim=-1,
    )
    torch.testing.assert_close(model, legacy)
    torch.testing.assert_close(codec.decode(model), physical)


def test_ego_codec_canonical_strategy_round_trip():
    codec = EgoTrajectoryCodec()
    physical = torch.tensor(
        [[[2.0, 1.0, math.pi - 1e-5], [4.0, -1.0, -math.pi + 1e-5]]]
    )
    strategy = codec.model_to_strategy(codec.encode(physical))
    assert strategy.shape == (1, 2, 4)
    torch.testing.assert_close(
        torch.linalg.vector_norm(strategy[..., 2:4], dim=-1),
        torch.ones(1, 2),
    )
    recovered = codec.strategy_to_physical(strategy)
    torch.testing.assert_close(recovered, physical, atol=1e-5, rtol=1e-5)


def test_reasoner_initializes_level_zero_and_advances_synchronously():
    config = LevelKConfig(
        reasoning_level=2,
        max_reasoning_level=3,
        max_agents=3,
        horizon_steps=4,
        interval_length=0.5,
    )
    agents = _agents()
    reasoner = LevelKReasoner(config)
    state = reasoner.initialize(
        ego_velocity=torch.tensor([[4.0, 0.0], [3.0, 0.0]]),
        agents=agents,
    )
    assert state.current_level == 0
    level_zero = state.current_strategies
    torch.testing.assert_close(
        level_zero.ego_trajectory[0, :, 0],
        torch.tensor([2.0, 4.0, 6.0, 8.0]),
    )

    # Simulate one completed synchronous best-response level.
    ego_level_one = level_zero.ego_trajectory + 0.25
    other_level_one = level_zero.other_trajectories + 0.5
    other_level_one = other_level_one * agents.agent_mask[..., None, None]
    state.append(
        ego_strategy=ego_level_one,
        other_strategies=other_level_one,
    )
    assert state.current_level == 1
    torch.testing.assert_close(
        state.current_strategies.ego_trajectory,
        ego_level_one,
    )

    ego_level_two = ego_level_one + 0.25
    other_level_two = (other_level_one + 0.5) * agents.agent_mask[
        ..., None, None
    ]
    state.append(
        ego_strategy=ego_level_two,
        other_strategies=other_level_two,
    )
    reasoner.validate_complete(state)
    output = state.output()
    assert output.reasoning_level == 2
    assert len(output.ego_levels) == 3
    assert len(output.other_levels) == 3


def test_reasoner_k_zero_requires_no_advanced_level():
    config = LevelKConfig(
        reasoning_level=0,
        max_agents=3,
        horizon_steps=4,
    )
    state = LevelKReasoner(config).initialize(
        ego_velocity=torch.ones(2, 2),
        agents=_agents(),
    )
    LevelKReasoner(config).validate_complete(state)
    assert state.output().reasoning_level == 0


def test_reasoner_rejects_nonzero_padding_and_incomplete_depth():
    config = LevelKConfig(
        reasoning_level=2,
        max_agents=3,
        horizon_steps=4,
    )
    reasoner = LevelKReasoner(config)
    state = reasoner.initialize(
        ego_velocity=torch.ones(2, 2),
        agents=_agents(),
    )
    current = state.current_strategies
    invalid_other = current.other_trajectories.clone()
    invalid_other[1, 2, 0, 0] = 1.0
    with pytest.raises(ValueError, match="Padded"):
        state.append(
            ego_strategy=current.ego_trajectory,
            other_strategies=invalid_other,
        )
    with pytest.raises(ValueError, match="expected 2"):
        reasoner.validate_complete(state)


def test_reasoner_rejects_wrong_agent_count():
    config = LevelKConfig(max_agents=10)
    with pytest.raises(ValueError, match="Expected N=10"):
        LevelKReasoner(config).initialize(
            ego_velocity=torch.ones(2, 2),
            agents=_agents(num_agents=3),
        )


def test_reasoner_run_uses_only_completed_previous_level():
    config = LevelKConfig(
        reasoning_level=3,
        max_reasoning_level=3,
        max_agents=3,
        horizon_steps=4,
    )
    reasoner = LevelKReasoner(config)
    state = reasoner.initialize(
        ego_velocity=torch.ones(2, 2),
        agents=_agents(),
    )
    encoded_levels = []
    predicted_levels = []

    def encode(strategies, source_level):
        encoded_levels.append(
            (source_level, strategies.ego_trajectory[0, 0, 0].item())
        )
        tokens = torch.zeros(2, 4, 8)
        mask = torch.ones(2, 4, dtype=torch.bool)
        return StrategyTokenBatch(tokens=tokens, token_mask=mask)

    def predict(tokens, target_level):
        predicted_levels.append(target_level)
        current = state.current_strategies
        ego = current.ego_trajectory.clone()
        ego[..., 0] = float(target_level)
        other = current.other_trajectories.clone()
        other[..., 0] = float(target_level)
        other = other * state.agent_mask[..., None, None]
        return ego, other

    output = reasoner.run(
        state=state,
        encode_strategy=encode,
        predict_best_responses=predict,
    )
    assert [level for level, _ in encoded_levels] == [0, 1, 2]
    assert predicted_levels == [1, 2, 3]
    assert encoded_levels[1][1] == 1.0
    assert encoded_levels[2][1] == 2.0
    assert output.ego_levels[-1][0, 0, 0] == 3.0
