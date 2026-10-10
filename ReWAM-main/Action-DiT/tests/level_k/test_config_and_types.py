"""Regression tests for the Level-k configuration and tensor contracts."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from navsim.agents.videodrive.level_k import (
    AgentTrajectoryBatch,
    LevelKConfig,
    LevelZeroReferenceBatch,
    synchronize_level_k_model_config,
    validate_level_k_trajectory_sampling,
)


def test_level_zero_reference_batch_validates_causal_map_contract() -> None:
    references = LevelZeroReferenceBatch(
        ego_trajectory=torch.tensor(
            [[[1.0, 0.0, 0.0, 1.0], [2.0, 0.1, 0.1, 0.995]]]
        ),
        ego_reference_mask=torch.tensor([True]),
        other_trajectories=torch.tensor(
            [[[[0.5, 0.0, 0.0, 1.0], [1.0, 0.1, 0.1, 0.995]],
              [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]]
        ),
        other_reference_mask=torch.tensor([[True, False]]),
    )
    references.validate(
        agent_mask=torch.tensor([[True, False]]),
        horizon_steps=2,
    )


def test_level_zero_reference_batch_rejects_padded_reference() -> None:
    references = LevelZeroReferenceBatch(
        ego_trajectory=torch.tensor([[[1.0, 0.0, 0.0, 1.0]]]),
        ego_reference_mask=torch.tensor([True]),
        other_trajectories=torch.tensor([[[[1.0, 0.0, 0.0, 1.0]]]]),
        other_reference_mask=torch.tensor([[True]]),
    )
    with pytest.raises(ValueError, match="Padded agents"):
        references.validate(agent_mask=torch.tensor([[False]]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enabled", 1),
        ("oracle_agent_states", "true"),
        ("require_visualization_acceptance", 1),
        ("train_ego_action_expert_from_scratch", 1),
        ("train_other_agent_expert_from_scratch", "true"),
        ("initialize_level_k_modules_from_scratch", 1),
        ("freeze_video_transformer", 1),
        ("freeze_vae", "true"),
        ("freeze_text_encoder", 1),
        ("reasoning_level", 1.5),
        ("max_reasoning_level", 3.0),
        ("training_reasoning_level", True),
        ("max_agents", 10.5),
        ("horizon_steps", True),
        ("level_zero_route_search_depth", 30.0),
        ("agent_range", True),
        ("interval_length", "0.5"),
        ("level_zero_mode", 1),
        ("level_zero_fallback", False),
        ("ego_action_lr", "1e-5"),
        ("other_agent_lr", True),
        ("strategy_lr", "3e-5"),
        ("collision_loss_start_step", 1.5),
        ("collision_loss_ramp_steps", True),
        ("collision_loss_weight", "0.05"),
        ("collision_temperature", True),
        ("progress_loss_start_step", 1.5),
        ("progress_loss_ramp_steps", True),
        ("progress_loss_weight", "0.1"),
        ("drivable_swept_substeps", True),
        ("drivable_worst_case_topk", 1.5),
        ("progress_safe_reference_gate", 1),
        ("hard_negative_enabled", 1),
        ("hard_negative_loss_start_step", 1.5),
        ("hard_negative_loss_ramp_steps", True),
        ("safe_trajectory_loss_start_step", 1.5),
        ("safe_trajectory_loss_ramp_steps", True),
        ("safe_trajectory_loss_weight", "0.1"),
        ("safe_trajectory_heading_weight", True),
        ("hard_negative_loss_weight", "0.02"),
        ("hard_negative_margin_ratio", True),
        ("hard_negative_max_margin", True),
        ("hard_negative_data_path", 123),
        ("hard_negative_split_path", 123),
    ],
)
def test_config_rejects_wrong_scalar_types(field: str, value) -> None:
    with pytest.raises(TypeError, match=field):
        LevelKConfig(**{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "agent_range",
        "interval_length",
        "level_zero_lane_search_radius",
        "level_zero_max_lateral_error",
        "level_zero_max_heading_error",
        "strategy_velocity_scale",
        "strategy_size_scale",
        "other_trajectory_position_scale",
        "ego_loss_weight",
        "other_loss_weight",
        "ego_action_lr",
        "other_agent_lr",
        "strategy_lr",
        "collision_loss_weight",
        "collision_safety_margin",
        "collision_temperature",
        "collision_radius_scale",
        "ego_vehicle_length",
        "ego_vehicle_width",
        "ego_rear_axle_to_center",
        "progress_loss_weight",
        "progress_tolerance_ratio",
        "progress_min_gt_distance",
        "safe_trajectory_loss_weight",
        "safe_trajectory_heading_weight",
        "hard_negative_loss_weight",
        "hard_negative_margin_ratio",
        "hard_negative_max_margin",
        "hard_negative_min_coverage",
    ],
)
def test_config_rejects_non_finite_numeric_values(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        LevelKConfig(**{field: float("nan")})


@pytest.mark.parametrize(
    "field",
    ["ego_action_lr", "other_agent_lr", "strategy_lr"],
)
def test_config_rejects_non_positive_learning_rates(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        LevelKConfig(**{field: 0.0})


def test_scratch_training_defaults_preserve_legacy_behavior() -> None:
    config = LevelKConfig()
    assert config.train_ego_action_expert_from_scratch is False
    assert config.train_other_agent_expert_from_scratch is False
    assert config.initialize_level_k_modules_from_scratch is True
    assert config.freeze_video_transformer is True
    assert config.freeze_vae is True
    assert config.freeze_text_encoder is True
    assert config.ego_action_lr == pytest.approx(1.0e-5)
    assert config.other_agent_lr == pytest.approx(3.0e-5)
    assert config.strategy_lr == pytest.approx(3.0e-5)
    assert config.collision_loss_weight == 0.0
    assert config.collision_safety_margin == pytest.approx(0.3)
    assert config.collision_temperature == pytest.approx(0.2)
    assert config.collision_radius_scale == pytest.approx(0.7)
    assert config.progress_loss_weight == 0.0
    assert config.progress_tolerance_ratio == pytest.approx(0.05)
    assert config.progress_min_gt_distance == pytest.approx(0.5)
    assert config.safe_trajectory_loss_weight == pytest.approx(0.1)
    assert config.safe_trajectory_heading_weight == pytest.approx(0.1)
    assert config.safe_trajectory_loss_start_step == 1000
    assert config.safe_trajectory_loss_ramp_steps == 4000
    assert config.hard_negative_enabled is False
    assert config.hard_negative_data_path is None
    assert config.hard_negative_split_path is None
    assert config.hard_negative_loss_weight == pytest.approx(0.02)
    assert config.hard_negative_margin_ratio == pytest.approx(0.5)
    assert config.hard_negative_max_margin == pytest.approx(2.0)
    assert config.hard_negative_loss_start_step == 2000
    assert config.hard_negative_loss_ramp_steps == 8000
    assert config.hard_negative_min_coverage == pytest.approx(0.8)
    assert config.safety_hard_enabled is False
    assert config.safety_hard_data_path is None
    assert config.safety_hard_sample_ratio == 0.0
    assert config.level_zero_mode == "constant_heading"
    assert config.level_zero_fallback == "constant_heading"
    assert config.level_zero_lane_search_radius == pytest.approx(8.0)
    assert config.level_zero_max_lateral_error == pytest.approx(4.0)
    assert config.level_zero_max_heading_error == pytest.approx(
        math.pi / 3.0
    )
    assert config.level_zero_route_search_depth == 30


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("level_zero_mode", "unknown"),
        ("level_zero_fallback", "none"),
        ("level_zero_lane_search_radius", 0.0),
        ("level_zero_max_lateral_error", 0.0),
        ("level_zero_max_heading_error", 0.0),
        ("level_zero_max_heading_error", 3.15),
        ("level_zero_route_search_depth", 0),
    ],
)
def test_level_zero_config_rejects_invalid_values(field: str, value) -> None:
    with pytest.raises(ValueError, match=field):
        LevelKConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("collision_loss_weight", -0.1),
        ("collision_safety_margin", -0.1),
        ("collision_temperature", 0.0),
        ("collision_radius_scale", 0.0),
        ("collision_loss_start_step", -1),
        ("collision_loss_ramp_steps", 0),
        ("ego_vehicle_length", 0.0),
        ("ego_vehicle_width", 0.0),
        ("ego_rear_axle_to_center", -0.1),
    ],
)
def test_collision_config_rejects_invalid_values(field: str, value) -> None:
    with pytest.raises(ValueError, match=field):
        LevelKConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("progress_loss_weight", -0.1),
        ("progress_tolerance_ratio", -0.1),
        ("progress_tolerance_ratio", 1.0),
        ("progress_min_gt_distance", -0.1),
        ("progress_loss_start_step", -1),
        ("progress_loss_ramp_steps", 0),
    ],
)
def test_progress_config_rejects_invalid_values(field: str, value) -> None:
    with pytest.raises(ValueError, match=field):
        LevelKConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("safe_trajectory_loss_weight", -0.1),
        ("safe_trajectory_heading_weight", -0.1),
        ("safe_trajectory_loss_start_step", -1),
        ("safe_trajectory_loss_ramp_steps", 0),
        ("hard_negative_loss_weight", -0.1),
        ("hard_negative_margin_ratio", -0.1),
        ("hard_negative_margin_ratio", 1.1),
        ("hard_negative_max_margin", 0.0),
        ("hard_negative_loss_start_step", -1),
        ("hard_negative_loss_ramp_steps", 0),
        ("hard_negative_min_coverage", -0.1),
        ("hard_negative_min_coverage", 1.1),
    ],
)
def test_safe_and_hard_negative_config_rejects_invalid_values(
    field: str, value
) -> None:
    with pytest.raises(ValueError, match=field):
        LevelKConfig(**{field: value})


def test_safety_hard_sampling_requires_manifest_and_enabled_level_k() -> None:
    with pytest.raises(ValueError, match="requires safety-hard sampling"):
        LevelKConfig(safety_hard_sample_ratio=0.35)
    with pytest.raises(ValueError, match="requires Level-k"):
        LevelKConfig(
            safety_hard_enabled=True,
            safety_hard_data_path="pool",
            safety_hard_sample_ratio=0.35,
        )
    config = LevelKConfig(
        enabled=True,
        safety_hard_enabled=True,
        safety_hard_data_path="pool",
        safety_hard_sample_ratio=0.35,
    )
    assert config.safety_hard_sample_ratio == pytest.approx(0.35)


def test_hard_negative_sampling_requires_enabled_supervision() -> None:
    with pytest.raises(ValueError, match="sample_ratio requires"):
        LevelKConfig(hard_negative_sample_ratio=0.15)

    config = LevelKConfig(
        enabled=True,
        hard_negative_enabled=True,
        hard_negative_data_path="pool",
        hard_negative_sample_ratio=0.15,
    )
    assert config.hard_negative_sample_ratio == pytest.approx(0.15)

    for invalid_ratio in (-0.1, 1.0):
        with pytest.raises(ValueError, match="hard_negative_sample_ratio"):
            LevelKConfig(hard_negative_sample_ratio=invalid_ratio)


def _base_model_config() -> dict:
    return {
        "action_expert": True,
        "action_strategy_conditioning": False,
        "other_agent_expert": False,
        "other_agent_horizon_steps": 4,
        "other_agent_position_scale": 20.0,
        "other_agent_velocity_scale": 10.0,
        "other_agent_size_scale": 5.0,
        "strategy_max_reasoning_level": 2,
        "strategy_interval_length": 1.0,
    }


def test_disabled_level_k_is_exact_model_config_no_op() -> None:
    model_config = _base_model_config()
    original = copy.deepcopy(model_config)
    synchronize_level_k_model_config(LevelKConfig(), model_config)
    assert model_config == original


def test_enabled_level_k_activates_and_aligns_transformer_modules() -> None:
    model_config = _base_model_config()
    config = LevelKConfig(
        enabled=True,
        max_reasoning_level=4,
        training_reasoning_level=2,
        horizon_steps=6,
        interval_length=0.25,
        strategy_velocity_scale=40.0,
        strategy_size_scale=12.0,
        other_trajectory_position_scale=100.0,
    )
    synchronize_level_k_model_config(config, model_config)
    assert model_config["action_strategy_conditioning"] is True
    assert model_config["other_agent_expert"] is True
    assert model_config["other_agent_horizon_steps"] == 6
    assert model_config["other_agent_position_scale"] == 100.0
    assert model_config["other_agent_velocity_scale"] == 40.0
    assert model_config["other_agent_size_scale"] == 12.0
    assert model_config["strategy_max_reasoning_level"] == 4
    assert model_config["strategy_interval_length"] == 0.25


def test_enabled_level_k_requires_existing_ego_action_expert() -> None:
    with pytest.raises(ValueError, match="action_expert=true"):
        synchronize_level_k_model_config(
            LevelKConfig(enabled=True),
            {"action_expert": False},
        )


def test_disabled_level_k_does_not_validate_trajectory_sampling() -> None:
    validate_level_k_trajectory_sampling(
        LevelKConfig(enabled=False),
        num_poses=99,
        interval_length=99.0,
    )


def test_enabled_level_k_accepts_matching_trajectory_sampling() -> None:
    validate_level_k_trajectory_sampling(
        LevelKConfig(
            enabled=True,
            horizon_steps=8,
            interval_length=0.5,
        ),
        num_poses=8,
        interval_length=0.5,
    )


@pytest.mark.parametrize(
    ("num_poses", "interval_length", "message"),
    [
        (7, 0.5, "horizon_steps"),
        (8, 0.25, "interval_length"),
    ],
)
def test_enabled_level_k_rejects_mismatched_trajectory_sampling(
    num_poses: int,
    interval_length: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_level_k_trajectory_sampling(
            LevelKConfig(
                enabled=True,
                horizon_steps=8,
                interval_length=0.5,
            ),
            num_poses=num_poses,
            interval_length=interval_length,
        )


def _trajectory_batch() -> AgentTrajectoryBatch:
    current = torch.zeros(1, 2, 8)
    current[..., 3] = 1.0
    future = torch.zeros(1, 2, 3, 4)
    future[..., 3] = 1.0
    return AgentTrajectoryBatch(
        current_states=current,
        agent_types=torch.tensor([[0, -1]], dtype=torch.long),
        agent_mask=torch.tensor([[True, False]]),
        future_trajectories=future,
        future_mask=torch.tensor(
            [[[True, True, False], [False, False, False]]]
        ),
    )


def test_trajectory_contract_rejects_valid_future_nan() -> None:
    batch = _trajectory_batch()
    batch.future_trajectories[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        batch.validate()


def test_trajectory_contract_allows_masked_future_nan() -> None:
    batch = _trajectory_batch()
    batch.future_trajectories[0, 0, 2] = float("nan")
    batch.future_trajectories[0, 1] = float("nan")
    batch.validate(max_agents=2, horizon_steps=3)


def test_trajectory_contract_rejects_padded_agent_supervision() -> None:
    batch = _trajectory_batch()
    batch.future_mask[0, 1, 0] = True
    with pytest.raises(ValueError, match="Padded agents"):
        batch.validate()


def test_trajectory_contract_reuses_current_state_validation() -> None:
    batch = _trajectory_batch()
    batch.current_states[0, 0, 2:4] = 0.0
    with pytest.raises(ValueError, match="non-zero magnitude"):
        batch.validate()


def test_trajectory_contract_requires_one_float_dtype() -> None:
    batch = _trajectory_batch()
    batch.future_trajectories = batch.future_trajectories.double()
    with pytest.raises(TypeError, match="share one dtype"):
        batch.validate()


def test_drivable_config_defaults_to_disabled() -> None:
    config = LevelKConfig()
    assert config.drivable_loss_weight == 0.0
    assert config.drivable_data_path is None
    assert config.drivable_safety_margin == pytest.approx(0.2)
    assert config.drivable_temperature == pytest.approx(0.2)
    assert config.drivable_swept_substeps == 1
    assert config.drivable_worst_case_topk == 0
    assert config.drivable_loss_start_step == 1000
    assert config.drivable_loss_ramp_steps == 4000
    assert config.drivable_min_coverage == pytest.approx(1.0)


def test_enabled_drivable_loss_requires_level_k_and_cache_path() -> None:
    with pytest.raises(ValueError, match="requires Level-k"):
        LevelKConfig(
            drivable_loss_weight=0.1,
            drivable_data_path="cache",
        )
    with pytest.raises(ValueError, match="drivable_data_path"):
        LevelKConfig(enabled=True, drivable_loss_weight=0.1)
    config = LevelKConfig(
        enabled=True,
        drivable_loss_weight=0.1,
        drivable_data_path="cache",
    )
    assert config.drivable_data_path == "cache"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("drivable_loss_weight", -0.1),
        ("drivable_safety_margin", -0.1),
        ("drivable_temperature", 0.0),
        ("drivable_swept_substeps", 0),
        ("drivable_worst_case_topk", -1),
        ("drivable_loss_start_step", -1),
        ("drivable_loss_ramp_steps", 0),
        ("drivable_min_coverage", -0.1),
        ("drivable_min_coverage", 1.1),
        ("drivable_data_path", 123),
    ],
)
def test_drivable_config_rejects_invalid_values(field: str, value) -> None:
    with pytest.raises((TypeError, ValueError)):
        LevelKConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("turn_min_reference_angle", -0.1, "angle thresholds"),
        ("turn_tolerance", 3.2, "angle thresholds"),
        ("turn_risk_margin", -0.1, "risk_margin"),
        ("turn_risk_temperature", 0.0, "risk_temperature"),
        ("turn_lateral_margin", -0.1, "lateral_margin"),
        ("turn_time_weight_min", 0.0, "time_weight_min"),
    ],
)
def test_turn_route_configuration_rejects_invalid_geometry(
    field: str,
    value: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        LevelKConfig(**{field: value})


def test_turn_route_configuration_requires_active_component() -> None:
    with pytest.raises(ValueError, match="at least one component"):
        LevelKConfig(
            enabled=True,
            turn_route_loss_weight=0.1,
            turn_component_weight=0.0,
            turn_heading_weight=0.0,
            turn_lateral_weight=0.0,
        )
