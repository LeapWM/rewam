"""Tests for deterministic Level-0 game policies."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from navsim.agents.videodrive.surround_agent_features import (
    SurroundAgentFeatureBuilder,
)
from navsim.agents.videodrive.level_k import (
    AgentStateBatch,
    AgentTrajectoryBatch,
    LevelZeroPolicy,
    LevelZeroReferenceBatch,
)


def _agents(
    *,
    batch_size: int = 1,
    num_agents: int = 10,
    dtype: torch.dtype = torch.float32,
) -> AgentStateBatch:
    states = torch.zeros(batch_size, num_agents, 8, dtype=dtype)
    states[..., 3] = 1.0
    states[..., 6:8] = torch.tensor([4.5, 2.0], dtype=dtype)
    types = torch.zeros(batch_size, num_agents, dtype=torch.int64)
    mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
    return AgentStateBatch(states, types, mask)


def test_time_sampling_and_ego_constant_speed() -> None:
    policy = LevelZeroPolicy(horizon_steps=8, interval_length=0.5)
    agents = _agents()
    output = policy(torch.tensor([[10.0, 2.0]]), agents)

    expected_time = torch.arange(1, 9, dtype=torch.float32) * 0.5
    torch.testing.assert_close(output.ego_trajectory[0, :, 0], 10.0 * expected_time)
    torch.testing.assert_close(
        output.ego_trajectory[0, :, 1],
        torch.zeros(8),
    )
    torch.testing.assert_close(
        output.ego_trajectory[0, :, 2],
        torch.zeros(8),
    )
    torch.testing.assert_close(
        output.ego_trajectory[0, :, 3],
        torch.ones(8),
    )


def test_other_agent_follows_heading_in_anchor_frame() -> None:
    policy = LevelZeroPolicy(horizon_steps=8, interval_length=0.5)
    agents = _agents()
    agents.current_states[0, 0, 2:4] = torch.tensor([1.0, 0.0])
    agents.current_states[0, 0, 4:6] = torch.tensor([3.0, 4.0])

    output = policy(torch.zeros(1, 2), agents)
    expected_time = torch.arange(1, 9, dtype=torch.float32) * 0.5
    torch.testing.assert_close(
        output.other_trajectories[0, 0, :, 0],
        torch.zeros(8),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output.other_trajectories[0, 0, :, 1],
        4.0 * expected_time,
    )
    torch.testing.assert_close(
        output.other_trajectories[0, 0, :, 2:],
        torch.tensor([1.0, 0.0]).expand(8, 2),
    )


def test_map_references_replace_only_valid_complete_paths() -> None:
    policy = LevelZeroPolicy(horizon_steps=2, interval_length=0.5)
    agents = _agents(num_agents=2)
    agents.current_states[..., 4] = 2.0
    references = LevelZeroReferenceBatch(
        ego_trajectory=torch.tensor(
            [[[0.8, 0.2, 0.1, 0.995], [1.6, 0.5, 0.2, 0.98]]]
        ),
        ego_reference_mask=torch.tensor([True]),
        other_trajectories=torch.tensor(
            [[
                [[0.9, 0.1, 0.1, 0.995], [1.8, 0.4, 0.2, 0.98]],
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            ]]
        ),
        other_reference_mask=torch.tensor([[True, False]]),
    )
    output = policy(torch.tensor([[3.0, 0.0]]), agents, references)

    torch.testing.assert_close(output.ego_trajectory, references.ego_trajectory)
    torch.testing.assert_close(
        output.other_trajectories[:, 0],
        references.other_trajectories[:, 0],
    )
    torch.testing.assert_close(
        output.other_trajectories[0, 1],
        torch.tensor(
            [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]]
        ),
    )


def test_level_zero_without_references_preserves_legacy_behavior() -> None:
    policy = LevelZeroPolicy(horizon_steps=2, interval_length=0.5)
    agents = _agents(num_agents=1)
    agents.current_states[..., 4] = 2.0
    legacy = policy(torch.tensor([[3.0, 0.0]]), agents)
    explicit_none = policy(torch.tensor([[3.0, 0.0]]), agents, None)
    torch.testing.assert_close(legacy.ego_trajectory, explicit_none.ego_trajectory)
    torch.testing.assert_close(
        legacy.other_trajectories,
        explicit_none.other_trajectories,
    )


def test_lateral_velocity_noise_is_not_integrated() -> None:
    policy = LevelZeroPolicy(horizon_steps=2, interval_length=0.5)
    agents = _agents()
    agents.current_states[0, 0, 4:6] = torch.tensor([6.0, 4.0])

    output = policy(torch.zeros(1, 2), agents)
    expected = torch.tensor(
        [[3.0, 0.0, 0.0, 1.0], [6.0, 0.0, 0.0, 1.0]]
    )
    torch.testing.assert_close(output.other_trajectories[0, 0], expected)


@pytest.mark.parametrize("num_agents", [8, 10])
def test_batch_shape_padding_and_mask(num_agents: int) -> None:
    policy = LevelZeroPolicy(horizon_steps=8, interval_length=0.5)
    agents = _agents(batch_size=2, num_agents=num_agents)
    agents.current_states[..., 4] = 5.0
    agents.agent_mask[:, -2:] = False

    output = policy(torch.tensor([[5.0, 0.0], [8.0, 0.0]]), agents)
    assert output.ego_trajectory.shape == (2, 8, 4)
    assert output.other_trajectories.shape == (2, num_agents, 8, 4)
    assert output.agent_mask.shape == (2, num_agents)
    assert torch.count_nonzero(output.other_trajectories[:, -2:]) == 0


def test_matches_straight_line_target_representation() -> None:
    policy = LevelZeroPolicy(horizon_steps=3, interval_length=0.5)
    agents = _agents()
    sin_heading = torch.tensor(0.6)
    cos_heading = torch.tensor(0.8)
    agents.current_states[0, 0, 2:4] = torch.stack((sin_heading, cos_heading))
    agents.current_states[0, 0, 4:6] = torch.tensor([4.0, 3.0])

    output = policy(torch.zeros(1, 2), agents)
    speed = 4.0 * cos_heading + 3.0 * sin_heading
    times = torch.tensor([0.5, 1.0, 1.5])
    expected = torch.stack(
        (
            speed * cos_heading * times,
            speed * sin_heading * times,
            sin_heading.expand(3),
            cos_heading.expand(3),
        ),
        dim=-1,
    )
    torch.testing.assert_close(output.other_trajectories[0, 0], expected)


def test_current_annotation_inputs_do_not_read_future_frames() -> None:
    annotations = SimpleNamespace(
        names=["vehicle"],
        track_tokens=["vehicle-1"],
        boxes=np.asarray([[10.0, 2.0, 0.2, 4.5, 2.0, 1.5, 0.0]]),
        velocity_3d=np.asarray([[5.0, 1.0, 0.0]]),
    )
    scene = SimpleNamespace(
        scene_metadata=SimpleNamespace(num_history_frames=1),
        frames=[
            SimpleNamespace(
                annotations=annotations,
                ego_status=SimpleNamespace(
                    ego_pose=np.asarray([0.0, 0.0, 0.0])
                ),
            )
        ],
    )
    current = SurroundAgentFeatureBuilder(
        max_agents=3,
        horizon_steps=8,
    ).compute_current_inputs(scene)
    assert current["agent_current_states"].shape == (3, 8)
    assert current["agent_mask"].tolist() == [True, False, False]
    torch.testing.assert_close(
        current["agent_current_states"][0, :2],
        torch.tensor([10.0, 2.0]),
    )


def test_negative_longitudinal_speed_becomes_stationary() -> None:
    policy = LevelZeroPolicy(horizon_steps=2, interval_length=0.5)
    agents = _agents()
    agents.current_states[0, 0, 4] = -3.0

    output = policy(torch.tensor([[-2.0, 0.0]]), agents)
    torch.testing.assert_close(
        output.ego_trajectory[0, :, :2],
        torch.zeros(2, 2),
    )
    torch.testing.assert_close(
        output.other_trajectories[0, 0, :, :2],
        torch.zeros(2, 2),
    )


def test_current_only_view_does_not_require_future_values() -> None:
    trajectory_batch = AgentTrajectoryBatch(
        current_states=_agents().current_states,
        agent_types=_agents().agent_types,
        agent_mask=_agents().agent_mask,
        future_trajectories=torch.full((1, 10, 8, 4), float("nan")),
        future_mask=torch.zeros(1, 10, 8, dtype=torch.bool),
    )

    current = trajectory_batch.current_inputs()
    current.validate(max_agents=10)
    LevelZeroPolicy()(torch.zeros(1, 2), current)


def test_dtype_and_determinism() -> None:
    policy = LevelZeroPolicy()
    agents = _agents(dtype=torch.float64)
    agents.current_states[..., 4] = 7.0
    velocity = torch.tensor([[7.0, 0.0]], dtype=torch.float64)

    first = policy(velocity, agents)
    second = policy(velocity, agents)
    assert first.ego_trajectory.dtype == torch.float64
    assert first.other_trajectories.dtype == torch.float64
    torch.testing.assert_close(first.ego_trajectory, second.ego_trajectory)
    torch.testing.assert_close(first.other_trajectories, second.other_trajectories)
    assert sum(parameter.numel() for parameter in policy.parameters()) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_dtype_is_preserved(dtype: torch.dtype) -> None:
    policy = LevelZeroPolicy(horizon_steps=2, interval_length=0.5)
    agents = _agents(dtype=dtype)
    agents.current_states[..., 4] = 3.0

    output = policy(torch.tensor([[3.0, 0.0]], dtype=dtype), agents)

    assert output.ego_trajectory.dtype == dtype
    assert output.other_trajectories.dtype == dtype
    assert torch.isfinite(output.ego_trajectory).all()
    assert torch.isfinite(output.other_trajectories).all()


def test_padded_non_finite_states_produce_exact_zero_trajectories() -> None:
    policy = LevelZeroPolicy(horizon_steps=3, interval_length=0.5)
    agents = _agents(batch_size=2, num_agents=3)
    agents.agent_mask[:, -1] = False
    agents.agent_types[:, -1] = -1
    agents.current_states[0, -1] = float("nan")
    agents.current_states[1, -1] = float("inf")

    output = policy(torch.tensor([[3.0, 0.0], [4.0, 0.0]]), agents)

    assert torch.isfinite(output.other_trajectories).all()
    assert torch.count_nonzero(output.other_trajectories[:, -1]) == 0


def test_empty_agent_batch_is_supported() -> None:
    policy = LevelZeroPolicy(horizon_steps=3, interval_length=0.5)
    agents = _agents(batch_size=2, num_agents=0)

    output = policy(torch.tensor([[3.0, 0.0], [4.0, 0.0]]), agents)

    assert output.other_trajectories.shape == (2, 0, 3, 4)
    assert torch.isfinite(output.ego_trajectory).all()
    assert torch.isfinite(output.other_trajectories).all()


def test_non_unit_heading_is_normalized_before_rollout() -> None:
    policy = LevelZeroPolicy(horizon_steps=2, interval_length=0.5)
    agents = _agents(num_agents=1)
    agents.current_states[0, 0, 2:4] = torch.tensor([6.0, 8.0])
    agents.current_states[0, 0, 4:6] = torch.tensor([4.0, 3.0])

    output = policy(torch.zeros(1, 2), agents)

    expected = torch.tensor(
        [
            [2.0, 1.5, 0.6, 0.8],
            [4.0, 3.0, 0.6, 0.8],
        ]
    )
    torch.testing.assert_close(output.other_trajectories[0, 0], expected)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"horizon_steps": 2.5}, TypeError),
        ({"horizon_steps": True}, TypeError),
        ({"interval_length": True}, TypeError),
        ({"interval_length": "0.5"}, TypeError),
        ({"interval_length": float("nan")}, ValueError),
        ({"interval_length": float("inf")}, ValueError),
        ({"heading_epsilon": True}, TypeError),
        ({"heading_epsilon": float("nan")}, ValueError),
        ({"heading_epsilon": float("inf")}, ValueError),
    ],
)
def test_constructor_rejects_invalid_scalar_values(
    kwargs: dict,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        LevelZeroPolicy(**kwargs)


def test_rejects_invalid_inputs() -> None:
    policy = LevelZeroPolicy()
    agents = _agents()

    with pytest.raises(ValueError, match="shape"):
        policy(torch.zeros(1, 3), agents)

    invalid_dtype = _agents()
    with pytest.raises(TypeError, match="dtype"):
        policy(torch.zeros(1, 2, dtype=torch.float64), invalid_dtype)

    invalid_nan = _agents()
    invalid_nan.current_states[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        policy(torch.zeros(1, 2), invalid_nan)

    invalid_heading = _agents()
    invalid_heading.current_states[0, 0, 2:4] = 0.0
    with pytest.raises(ValueError, match="heading"):
        policy(torch.zeros(1, 2), invalid_heading)
