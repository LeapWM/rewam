"""Tests for the shared surrounding-agent Flow-Matching expert."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from navsim.agents.videodrive.level_k import (
    AgentStateBatch,
    OtherTrajectoryCodec,
    SharedOtherAgentExpert,
    StrategyTokenBatch,
    select_video_layer_indices,
)


def _expert(
    *,
    horizon_steps: int = 4,
    hidden_dim: int = 32,
    video_hidden_dim: int = 64,
    num_layers: int = 2,
    num_video_layers: int = 4,
) -> SharedOtherAgentExpert:
    return SharedOtherAgentExpert(
        horizon_steps=horizon_steps,
        hidden_dim=hidden_dim,
        video_hidden_dim=video_hidden_dim,
        num_heads=4,
        head_dim=hidden_dim // 4,
        num_layers=num_layers,
        num_video_layers=num_video_layers,
    )


def _inputs(
    *,
    batch_size: int = 1,
    num_agents: int = 3,
    horizon_steps: int = 4,
    hidden_dim: int = 32,
    video_hidden_dim: int = 64,
    num_video_layers: int = 4,
    dtype: torch.dtype = torch.float32,
):
    generator = torch.Generator().manual_seed(7)
    states = torch.randn(
        batch_size,
        num_agents,
        8,
        generator=generator,
        dtype=dtype,
    )
    states[..., 2] = 0.0
    states[..., 3] = 1.0
    states[..., 6:8] = torch.tensor([4.5, 2.0], dtype=dtype)
    mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
    agents = AgentStateBatch(
        current_states=states,
        agent_types=torch.zeros(batch_size, num_agents, dtype=torch.long),
        agent_mask=mask,
    )
    token_mask = torch.cat(
        (torch.ones(batch_size, 1, dtype=torch.bool), mask),
        dim=1,
    )
    tokens = torch.randn(
        batch_size,
        num_agents + 1,
        hidden_dim,
        generator=generator,
        dtype=dtype,
    )
    strategies = StrategyTokenBatch(tokens=tokens, token_mask=token_mask)
    noisy = torch.randn(
        batch_size,
        num_agents,
        horizon_steps,
        4,
        generator=generator,
        dtype=dtype,
    )
    video = [
        torch.randn(
            batch_size,
            5,
            video_hidden_dim,
            generator=generator,
            dtype=dtype,
        )
        for _ in range(num_video_layers)
    ]
    return noisy, agents, strategies, video


def _forward(
    model: SharedOtherAgentExpert,
    inputs,
    *,
    timestep=100.0,
):
    noisy, agents, strategies, video = inputs
    return model(
        noisy_trajectories=noisy,
        timestep=timestep,
        agent_states=agents,
        strategy_tokens=strategies,
        video_hidden_states=video,
    )


def test_video_layer_selection() -> None:
    assert select_video_layer_indices(
        num_expert_layers=1,
        num_video_layers=28,
    ) == (27,)
    assert select_video_layer_indices(
        num_expert_layers=4,
        num_video_layers=28,
    ) == (0, 9, 18, 27)
    assert select_video_layer_indices(
        num_expert_layers=8,
        num_video_layers=28,
    ) == (0, 4, 8, 12, 15, 19, 23, 27)
    assert select_video_layer_indices(
        num_expert_layers=14,
        num_video_layers=28,
    ) == (0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 21, 23, 25, 27)
    assert select_video_layer_indices(
        num_expert_layers=28,
        num_video_layers=28,
    ) == tuple(range(28))
    with pytest.raises(ValueError, match="cannot exceed"):
        select_video_layer_indices(
            num_expert_layers=29,
            num_video_layers=28,
        )
    with pytest.raises(TypeError, match="num_expert_layers"):
        select_video_layer_indices(
            num_expert_layers=True,
            num_video_layers=28,
        )
    with pytest.raises(TypeError, match="num_video_layers"):
        select_video_layer_indices(
            num_expert_layers=8,
            num_video_layers=28.0,
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("horizon_steps", True, TypeError),
        ("horizon_steps", 8.0, TypeError),
        ("input_dim", True, TypeError),
        ("output_dim", 4.0, TypeError),
        ("hidden_dim", True, TypeError),
        ("video_hidden_dim", 64.0, TypeError),
        ("num_heads", True, TypeError),
        ("head_dim", 8.0, TypeError),
        ("num_layers", True, TypeError),
        ("num_video_layers", 4.0, TypeError),
        ("num_layers", 0, ValueError),
        ("state_position_scale", True, TypeError),
        ("state_position_scale", float("nan"), ValueError),
        ("state_velocity_scale", float("inf"), ValueError),
        ("state_size_scale", float("-inf"), ValueError),
    ],
)
def test_constructor_rejects_invalid_scalar_values(
    field: str,
    value,
    error: type[Exception],
) -> None:
    values = {
        "horizon_steps": 4,
        "input_dim": 4,
        "output_dim": 4,
        "hidden_dim": 32,
        "video_hidden_dim": 64,
        "num_heads": 4,
        "head_dim": 8,
        "num_layers": 2,
        "num_video_layers": 4,
        "state_position_scale": 80.0,
        "state_velocity_scale": 30.0,
        "state_size_scale": 10.0,
    }
    values[field] = value
    with pytest.raises(error, match=field):
        SharedOtherAgentExpert(**values)


@pytest.mark.parametrize("num_agents", [8, 10])
def test_shape_for_eight_or_ten_agents(num_agents: int) -> None:
    model = _expert()
    output = _forward(model, _inputs(num_agents=num_agents))
    assert output.shape == (1, num_agents, 4, 4)


def test_parameter_count_is_independent_of_agent_count() -> None:
    model = _expert()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    _forward(model, _inputs(num_agents=8))
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count
    _forward(model, _inputs(num_agents=10))
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count


def test_each_selected_video_layer_is_consumed_once() -> None:
    model = _expert(num_layers=2, num_video_layers=4)
    inputs = _inputs(num_video_layers=4)
    call_counts = [0, 0]
    hooks = []
    for index, block in enumerate(model.blocks):
        hooks.append(
            block.register_forward_hook(
                lambda _module, _args, _output, index=index: call_counts.__setitem__(
                    index, call_counts[index] + 1
                )
            )
        )
    try:
        _forward(model, inputs)
    finally:
        for hook in hooks:
            hook.remove()
    assert model.video_layer_indices == (0, 3)
    assert call_counts == [1, 1]


def test_twenty_eight_layer_expert_consumes_every_video_layer_and_backpropagates() -> None:
    model = _expert(
        horizon_steps=2,
        hidden_dim=16,
        video_hidden_dim=16,
        num_layers=28,
        num_video_layers=28,
    )
    inputs = _inputs(
        num_agents=2,
        horizon_steps=2,
        hidden_dim=16,
        video_hidden_dim=16,
        num_video_layers=28,
    )
    call_counts = [0] * 28
    hooks = []
    for index, block in enumerate(model.blocks):
        hooks.append(
            block.register_forward_hook(
                lambda _module, _args, _output, index=index: call_counts.__setitem__(
                    index, call_counts[index] + 1
                )
            )
        )
    try:
        output = _forward(model, inputs)
    finally:
        for hook in hooks:
            hook.remove()

    assert output.shape == (1, 2, 2, 4)
    assert model.video_layer_indices == tuple(range(28))
    assert call_counts == [1] * 28

    output.square().mean().backward()
    assert model.blocks[0].strategy_attention.to_q.weight.grad is not None
    assert model.blocks[-1].strategy_attention.to_q.weight.grad is not None
    assert model.blocks[0].video_attention.to_k.weight.grad is not None
    assert model.blocks[-1].video_attention.to_k.weight.grad is not None


def test_video_hidden_states_must_share_dtype_and_device() -> None:
    model = _expert(num_layers=1, num_video_layers=1)
    noisy, agents, strategies, video = _inputs(num_video_layers=1)

    wrong_dtype = [video[0].double()]
    with pytest.raises(TypeError, match="dtype"):
        _forward(model, (noisy, agents, strategies, wrong_dtype))

    wrong_device = [torch.empty_like(video[0], device="meta")]
    with pytest.raises(ValueError, match="device"):
        _forward(model, (noisy, agents, strategies, wrong_device))


def test_agent_permutation_equivariance() -> None:
    model = _expert().eval()
    noisy, agents, strategies, video = _inputs()
    original = _forward(model, (noisy, agents, strategies, video))
    permutation = torch.tensor([2, 0, 1])
    permuted_inputs = (
        noisy[:, permutation],
        AgentStateBatch(
            current_states=agents.current_states[:, permutation],
            agent_types=agents.agent_types[:, permutation],
            agent_mask=agents.agent_mask[:, permutation],
        ),
        StrategyTokenBatch(
            tokens=torch.cat(
                (
                    strategies.tokens[:, :1],
                    strategies.other_tokens[:, permutation],
                ),
                dim=1,
            ),
            token_mask=torch.cat(
                (
                    strategies.token_mask[:, :1],
                    strategies.other_mask[:, permutation],
                ),
                dim=1,
            ),
        ),
        video,
    )
    permuted = _forward(model, permuted_inputs)
    torch.testing.assert_close(permuted, original[:, permutation])


def test_focal_agent_cannot_read_its_own_strategy() -> None:
    model = _expert().eval()
    noisy, agents, strategies, video = _inputs()
    original = _forward(model, (noisy, agents, strategies, video))
    changed_tokens = strategies.tokens.clone()
    changed_tokens[:, 1] += 100.0
    changed = _forward(
        model,
        (
            noisy,
            agents,
            StrategyTokenBatch(changed_tokens, strategies.token_mask),
            video,
        ),
    )

    torch.testing.assert_close(changed[:, 0], original[:, 0])
    assert not torch.allclose(changed[:, 1:], original[:, 1:])


def test_other_agent_responds_to_ego_and_opponent_strategy() -> None:
    model = _expert().eval()
    noisy, agents, strategies, video = _inputs()
    original = _forward(model, (noisy, agents, strategies, video))

    ego_changed = strategies.tokens.clone()
    ego_changed[:, 0] += 50.0
    ego_output = _forward(
        model,
        (
            noisy,
            agents,
            StrategyTokenBatch(ego_changed, strategies.token_mask),
            video,
        ),
    )
    assert not torch.allclose(ego_output[:, 0], original[:, 0])

    opponent_changed = strategies.tokens.clone()
    opponent_changed[:, 2] -= 50.0
    opponent_output = _forward(
        model,
        (
            noisy,
            agents,
            StrategyTokenBatch(opponent_changed, strategies.token_mask),
            video,
        ),
    )
    assert not torch.allclose(opponent_output[:, 0], original[:, 0])


def test_no_same_level_cross_agent_leakage() -> None:
    model = _expert().eval()
    noisy, agents, strategies, video = _inputs()
    original = _forward(model, (noisy, agents, strategies, video))
    changed_noisy = noisy.clone()
    changed_noisy[:, 1] += 100.0
    changed = _forward(model, (changed_noisy, agents, strategies, video))

    torch.testing.assert_close(changed[:, 0], original[:, 0])
    assert not torch.allclose(changed[:, 1], original[:, 1])


def test_padding_and_empty_agent_set_are_zero_and_finite() -> None:
    model = _expert().eval()
    noisy, agents, strategies, video = _inputs()
    agents.agent_mask[:, 1:] = False
    strategies.token_mask[:, 2:] = False
    strategies.tokens[:, 2:] = 0.0
    noisy[:, 1:] = float("nan")
    agents.current_states[:, 1:] = float("nan")
    output = _forward(model, (noisy, agents, strategies, video))
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[:, 1:]) == 0

    agents.agent_mask[:] = False
    strategies.token_mask[:, 1:] = False
    strategies.tokens[:, 1:] = 0.0
    noisy[:] = float("nan")
    agents.current_states[:] = float("nan")
    empty_output = _forward(model, (noisy, agents, strategies, video))
    assert torch.isfinite(empty_output).all()
    assert torch.count_nonzero(empty_output) == 0


def test_timestep_state_and_gradients_affect_output() -> None:
    model = _expert()
    inputs = _inputs()
    noisy, agents, strategies, video = inputs
    original = _forward(model, inputs, timestep=10.0)
    different_time = _forward(model, inputs, timestep=900.0)
    assert not torch.allclose(original, different_time)

    changed_states = agents.current_states.clone()
    changed_states[:, 0, 4] += 10.0
    state_output = _forward(
        model,
        (
            noisy,
            AgentStateBatch(
                changed_states,
                agents.agent_types,
                agents.agent_mask,
            ),
            strategies,
            video,
        ),
        timestep=10.0,
    )
    assert not torch.allclose(original[:, 0], state_output[:, 0])

    loss = original.square().mean()
    loss.backward()
    assert model.input_projection.weight.grad is not None
    assert model.blocks[0].strategy_attention.to_q.weight.grad is not None
    assert model.blocks[0].video_attention.to_k.weight.grad is not None


def test_float64_dtype() -> None:
    model = _expert().to(dtype=torch.float64)
    output = _forward(model, _inputs(dtype=torch.float64))
    assert output.dtype == torch.float64


def test_actual_d512_n10_contract() -> None:
    model = SharedOtherAgentExpert(
        horizon_steps=8,
        hidden_dim=512,
        video_hidden_dim=2048,
        num_heads=16,
        head_dim=32,
        num_layers=1,
        num_video_layers=28,
    )
    output = _forward(
        model,
        _inputs(
            num_agents=10,
            horizon_steps=8,
            hidden_dim=512,
            video_hidden_dim=2048,
            num_video_layers=1,
        ),
    )
    assert output.shape == (1, 10, 8, 4)
    assert model.video_layer_indices == (27,)


def test_optional_module_checkpoint_compatibility() -> None:
    class HostModel(nn.Module):
        def __init__(self, enabled: bool) -> None:
            super().__init__()
            self.video_projection = nn.Linear(4, 4)
            if enabled:
                self.other_agent_model = _expert(
                    horizon_steps=2,
                    hidden_dim=16,
                    video_hidden_dim=32,
                    num_layers=1,
                    num_video_layers=1,
                )

    original = HostModel(enabled=False)
    original_state = original.state_dict()
    disabled_reload = HostModel(enabled=False)
    disabled_result = disabled_reload.load_state_dict(original_state, strict=True)
    assert disabled_result.missing_keys == []
    assert disabled_result.unexpected_keys == []

    enabled_reload = HostModel(enabled=True)
    enabled_result = enabled_reload.load_state_dict(original_state, strict=False)
    assert enabled_result.unexpected_keys == []
    assert enabled_result.missing_keys
    assert all(
        key.startswith("other_agent_model.")
        for key in enabled_result.missing_keys
    )
    torch.testing.assert_close(
        enabled_reload.video_projection.weight,
        original.video_projection.weight,
    )


def test_trajectory_codec_roundtrip_no_clipping_and_heading_normalization() -> None:
    codec = OtherTrajectoryCodec(position_scale=80.0)
    physical = torch.tensor(
        [[[[160.0, -120.0, 0.6, 0.8], [8.0, 4.0, 0.0, 2.0]]]]
    )
    mask = torch.ones(1, 1, dtype=torch.bool)
    encoded = codec.encode(physical, mask)
    assert encoded[0, 0, 0, 0] == 2.0
    assert encoded[0, 0, 0, 1] == -1.5
    decoded = codec.decode(encoded, mask)
    torch.testing.assert_close(decoded[..., :2], physical[..., :2])
    torch.testing.assert_close(
        torch.linalg.vector_norm(decoded[..., 2:4], dim=-1),
        torch.ones(1, 1, 2),
    )
    torch.testing.assert_close(decoded[0, 0, 0, 2:], physical[0, 0, 0, 2:])
    torch.testing.assert_close(decoded[0, 0, 1, 2:], torch.tensor([0.0, 1.0]))


def test_trajectory_codec_padding_and_validation() -> None:
    codec = OtherTrajectoryCodec()
    values = torch.zeros(1, 2, 3, 4)
    mask = torch.tensor([[True, False]])
    values[:, 1] = float("nan")
    encoded = codec.encode(values, mask)
    decoded = codec.decode(encoded, mask)
    assert torch.count_nonzero(encoded[:, 1]) == 0
    assert torch.count_nonzero(decoded[:, 1]) == 0
    with pytest.raises(ValueError, match="positive"):
        OtherTrajectoryCodec(position_scale=0.0)
    with pytest.raises(ValueError, match="shape"):
        codec.encode(torch.zeros(1, 3, 4), None)
