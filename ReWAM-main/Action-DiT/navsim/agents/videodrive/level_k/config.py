"""Configuration contract for the optional Level-k reasoning path."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Mapping, MutableMapping


@dataclass(frozen=True)
class LevelKConfig:
    """Validated Level-k settings with backward-compatible defaults."""

    enabled: bool = False
    reasoning_level: int = 0
    max_reasoning_level: int = 3
    training_reasoning_level: int = 1
    max_agents: int = 10
    agent_range: float = 80.0
    agent_selection: str = "nearest"
    horizon_steps: int = 8
    interval_length: float = 0.5
    level_zero_mode: str = "constant_heading"
    level_zero_fallback: str = "constant_heading"
    level_zero_lane_search_radius: float = 8.0
    level_zero_max_lateral_error: float = 4.0
    level_zero_max_heading_error: float = math.pi / 3.0
    level_zero_route_search_depth: int = 30
    oracle_agent_states: bool = True
    strategy_velocity_scale: float = 30.0
    strategy_size_scale: float = 10.0
    other_trajectory_position_scale: float = 80.0
    ego_loss_weight: float = 1.0
    other_loss_weight: float = 1.0
    collision_loss_weight: float = 0.0
    collision_safety_margin: float = 0.3
    collision_temperature: float = 0.2
    collision_radius_scale: float = 0.7
    collision_loss_start_step: int = 1000
    collision_loss_ramp_steps: int = 4000
    ttc_swept_box_loss_weight: float = 0.0
    ttc_swept_substeps: int = 4
    ttc_reaction_time: float = 0.5
    ttc_safety_margin: float = 0.2
    ttc_temperature: float = 0.2
    ttc_loss_start_step: int = 1000
    ttc_loss_ramp_steps: int = 4000
    drivable_loss_weight: float = 0.0
    drivable_data_path: str | None = None
    drivable_safety_margin: float = 0.2
    drivable_temperature: float = 0.2
    drivable_swept_substeps: int = 1
    drivable_worst_case_topk: int = 0
    drivable_loss_start_step: int = 1000
    drivable_loss_ramp_steps: int = 4000
    drivable_min_coverage: float = 1.0
    turn_route_loss_weight: float = 0.0
    turn_route_loss_start_step: int = 1000
    turn_route_loss_ramp_steps: int = 4000
    turn_min_reference_angle: float = math.radians(10.0)
    turn_tolerance: float = math.radians(5.0)
    turn_direction_consistency_min: float = 0.8
    turn_risk_margin: float = 0.3
    turn_risk_temperature: float = 0.15
    turn_lateral_margin: float = 0.4
    turn_time_weight_min: float = 0.25
    turn_component_weight: float = 1.0
    turn_heading_weight: float = 0.25
    turn_lateral_weight: float = 0.5
    progress_loss_weight: float = 0.0
    progress_tolerance_ratio: float = 0.05
    progress_min_gt_distance: float = 0.5
    progress_loss_start_step: int = 1000
    progress_loss_ramp_steps: int = 4000
    progress_safe_loss_weight: float = 0.0
    progress_route_data_path: str | None = None
    progress_route_min_coverage: float = 1.0
    progress_projection_temperature: float = 0.5
    progress_safe_margin: float = 0.2
    progress_safe_min_gt_progress: float = 0.5
    progress_drivable_gate_margin: float = 0.2
    progress_drivable_gate_temperature: float = 0.2
    progress_ttc_gate_margin: float = 0.2
    progress_ttc_gate_temperature: float = 0.2
    progress_safe_reference_gate: bool = False
    pdm_rollout_enabled: bool = False
    pdm_auxiliary_final_level_only: bool = True
    pdm_progress_tolerance_ratio: float = 0.02
    pdm_progress_distance_threshold: float = 5.0
    progress_safe_loss_start_step: int = 1000
    progress_safe_loss_ramp_steps: int = 4000
    retain_loss_weight: float = 0.0
    retain_data_path: str | None = None
    retain_min_coverage: float = 1.0
    retain_min_safety_score: float = 0.8
    retain_heading_weight: float = 0.1
    retain_loss_start_step: int = 1000
    retain_loss_ramp_steps: int = 4000
    safe_trajectory_loss_weight: float = 0.1
    safe_trajectory_heading_weight: float = 0.1
    safe_trajectory_loss_start_step: int = 1000
    safe_trajectory_loss_ramp_steps: int = 4000
    hard_negative_enabled: bool = False
    hard_negative_data_path: str | None = None
    hard_negative_split_path: str | None = None
    hard_negative_sample_ratio: float = 0.0
    hard_negative_loss_weight: float = 0.02
    hard_negative_margin_ratio: float = 0.5
    hard_negative_max_margin: float = 2.0
    hard_negative_loss_start_step: int = 2000
    hard_negative_loss_ramp_steps: int = 8000
    hard_negative_min_coverage: float = 0.8
    safety_hard_enabled: bool = False
    safety_hard_data_path: str | None = None
    safety_hard_sample_ratio: float = 0.0
    ego_vehicle_length: float = 5.176
    ego_vehicle_width: float = 2.297
    ego_rear_axle_to_center: float = 1.461
    # Historical names retained for compatibility; these flags select full
    # joint expert trainability, independently of the initialization policy.
    train_ego_action_expert_from_scratch: bool = False
    train_other_agent_expert_from_scratch: bool = False
    # A pretrained Level-k checkpoint can warm-start all optimizer groups by
    # disabling this reset while leaving both trainability flags enabled.
    initialize_level_k_modules_from_scratch: bool = True
    freeze_video_transformer: bool = True
    freeze_vae: bool = True
    freeze_text_encoder: bool = True
    ego_action_lr: float = 1.0e-5
    other_agent_lr: float = 3.0e-5
    strategy_lr: float = 3.0e-5
    validation_report_path: str | None = None
    require_visualization_acceptance: bool = True

    def __post_init__(self) -> None:
        boolean_fields = (
            "enabled",
            "oracle_agent_states",
            "require_visualization_acceptance",
            "train_ego_action_expert_from_scratch",
            "train_other_agent_expert_from_scratch",
            "initialize_level_k_modules_from_scratch",
            "freeze_video_transformer",
            "freeze_vae",
            "freeze_text_encoder",
            "progress_safe_reference_gate",
            "pdm_rollout_enabled",
            "pdm_auxiliary_final_level_only",
            "hard_negative_enabled",
            "safety_hard_enabled",
        )
        for name in boolean_fields:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must use a boolean value.")

        integer_fields = (
            "reasoning_level",
            "max_reasoning_level",
            "training_reasoning_level",
            "max_agents",
            "horizon_steps",
            "level_zero_route_search_depth",
            "collision_loss_start_step",
            "collision_loss_ramp_steps",
            "ttc_swept_substeps",
            "ttc_loss_start_step",
            "ttc_loss_ramp_steps",
            "drivable_swept_substeps",
            "drivable_worst_case_topk",
            "drivable_loss_start_step",
            "drivable_loss_ramp_steps",
            "turn_route_loss_start_step",
            "turn_route_loss_ramp_steps",
            "progress_loss_start_step",
            "progress_loss_ramp_steps",
            "progress_safe_loss_start_step",
            "progress_safe_loss_ramp_steps",
            "retain_loss_start_step",
            "retain_loss_ramp_steps",
            "safe_trajectory_loss_start_step",
            "safe_trajectory_loss_ramp_steps",
            "hard_negative_loss_start_step",
            "hard_negative_loss_ramp_steps",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must use an integer value.")

        real_fields = (
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
            "collision_loss_weight",
            "collision_safety_margin",
            "collision_temperature",
            "collision_radius_scale",
            "ttc_swept_box_loss_weight",
            "ttc_reaction_time",
            "ttc_safety_margin",
            "ttc_temperature",
            "drivable_loss_weight",
            "drivable_safety_margin",
            "drivable_temperature",
            "drivable_min_coverage",
            "turn_route_loss_weight",
            "turn_min_reference_angle",
            "turn_tolerance",
            "turn_direction_consistency_min",
            "turn_risk_margin",
            "turn_risk_temperature",
            "turn_lateral_margin",
            "turn_time_weight_min",
            "turn_component_weight",
            "turn_heading_weight",
            "turn_lateral_weight",
            "progress_loss_weight",
            "progress_tolerance_ratio",
            "progress_min_gt_distance",
            "progress_safe_loss_weight",
            "progress_route_min_coverage",
            "progress_projection_temperature",
            "progress_safe_margin",
            "progress_safe_min_gt_progress",
            "progress_drivable_gate_margin",
            "progress_drivable_gate_temperature",
            "progress_ttc_gate_margin",
            "progress_ttc_gate_temperature",
            "pdm_progress_tolerance_ratio",
            "pdm_progress_distance_threshold",
            "retain_loss_weight",
            "retain_min_coverage",
            "retain_min_safety_score",
            "retain_heading_weight",
            "safe_trajectory_loss_weight",
            "safe_trajectory_heading_weight",
            "hard_negative_loss_weight",
            "hard_negative_sample_ratio",
            "hard_negative_margin_ratio",
            "hard_negative_max_margin",
            "hard_negative_min_coverage",
            "safety_hard_sample_ratio",
            "ego_vehicle_length",
            "ego_vehicle_width",
            "ego_rear_axle_to_center",
            "ego_action_lr",
            "other_agent_lr",
            "strategy_lr",
        )
        for name in real_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must use a real numeric value.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")

        if self.max_reasoning_level < 0:
            raise ValueError("max_reasoning_level must be non-negative.")
        if not 0 <= self.reasoning_level <= self.max_reasoning_level:
            raise ValueError(
                "reasoning_level must satisfy "
                "0 <= reasoning_level <= max_reasoning_level."
            )
        if not 1 <= self.training_reasoning_level <= self.max_reasoning_level:
            raise ValueError(
                "training_reasoning_level must satisfy "
                "1 <= training_reasoning_level <= max_reasoning_level."
            )
        if self.max_agents <= 0:
            raise ValueError("max_agents must be positive.")
        if self.agent_range <= 0:
            raise ValueError("agent_range must be positive.")
        if not isinstance(self.agent_selection, str):
            raise TypeError("agent_selection must use a string value.")
        if self.agent_selection != "nearest":
            raise ValueError("agent_selection currently only supports 'nearest'.")
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive.")
        if self.interval_length <= 0:
            raise ValueError("interval_length must be positive.")
        if not isinstance(self.level_zero_mode, str):
            raise TypeError("level_zero_mode must use a string value.")
        if self.level_zero_mode not in {
            "constant_heading",
            "map_lane_following",
        }:
            raise ValueError(
                "level_zero_mode must be 'constant_heading' or "
                "'map_lane_following'."
            )
        if not isinstance(self.level_zero_fallback, str):
            raise TypeError("level_zero_fallback must use a string value.")
        if self.level_zero_fallback != "constant_heading":
            raise ValueError(
                "level_zero_fallback currently only supports "
                "'constant_heading'."
            )
        if self.level_zero_lane_search_radius <= 0:
            raise ValueError("level_zero_lane_search_radius must be positive.")
        if self.level_zero_max_lateral_error <= 0:
            raise ValueError("level_zero_max_lateral_error must be positive.")
        if not 0 < self.level_zero_max_heading_error <= math.pi:
            raise ValueError(
                "level_zero_max_heading_error must lie in (0, pi]."
            )
        if self.level_zero_route_search_depth <= 0:
            raise ValueError("level_zero_route_search_depth must be positive.")
        if self.strategy_velocity_scale <= 0:
            raise ValueError("strategy_velocity_scale must be positive.")
        if self.strategy_size_scale <= 0:
            raise ValueError("strategy_size_scale must be positive.")
        if self.other_trajectory_position_scale <= 0:
            raise ValueError("other_trajectory_position_scale must be positive.")
        if self.ego_loss_weight < 0:
            raise ValueError("ego_loss_weight must be non-negative.")
        if self.other_loss_weight < 0:
            raise ValueError("other_loss_weight must be non-negative.")
        if self.ego_loss_weight == 0 and self.other_loss_weight == 0:
            raise ValueError("At least one Level-k loss weight must be positive.")
        if self.collision_loss_weight < 0:
            raise ValueError("collision_loss_weight must be non-negative.")
        if self.collision_safety_margin < 0:
            raise ValueError("collision_safety_margin must be non-negative.")
        if self.collision_temperature <= 0:
            raise ValueError("collision_temperature must be positive.")
        if self.collision_radius_scale <= 0:
            raise ValueError("collision_radius_scale must be positive.")
        if self.collision_loss_start_step < 0:
            raise ValueError("collision_loss_start_step must be non-negative.")
        if self.collision_loss_ramp_steps <= 0:
            raise ValueError("collision_loss_ramp_steps must be positive.")
        if self.ttc_swept_box_loss_weight < 0:
            raise ValueError("ttc_swept_box_loss_weight must be non-negative.")
        if self.ttc_swept_substeps <= 0:
            raise ValueError("ttc_swept_substeps must be positive.")
        if self.ttc_reaction_time < 0 or self.ttc_safety_margin < 0:
            raise ValueError("TTC reaction time and margin must be non-negative.")
        if self.ttc_temperature <= 0:
            raise ValueError("ttc_temperature must be positive.")
        if self.ttc_loss_start_step < 0 or self.ttc_loss_ramp_steps <= 0:
            raise ValueError("Invalid TTC loss ramp configuration.")
        if self.drivable_loss_weight < 0:
            raise ValueError("drivable_loss_weight must be non-negative.")
        if self.drivable_safety_margin < 0:
            raise ValueError("drivable_safety_margin must be non-negative.")
        if self.drivable_temperature <= 0:
            raise ValueError("drivable_temperature must be positive.")
        if self.drivable_swept_substeps <= 0:
            raise ValueError("drivable_swept_substeps must be positive.")
        if self.drivable_worst_case_topk < 0:
            raise ValueError("drivable_worst_case_topk must be non-negative.")
        if self.drivable_loss_start_step < 0:
            raise ValueError("drivable_loss_start_step must be non-negative.")
        if self.drivable_loss_ramp_steps <= 0:
            raise ValueError("drivable_loss_ramp_steps must be positive.")
        if not 0.0 <= self.drivable_min_coverage <= 1.0:
            raise ValueError("drivable_min_coverage must lie in [0,1].")
        if self.drivable_data_path is not None and not isinstance(
            self.drivable_data_path,
            str,
        ):
            raise TypeError("drivable_data_path must be a string or None.")
        if self.drivable_loss_weight > 0:
            if not self.enabled:
                raise ValueError(
                    "A positive drivable_loss_weight requires Level-k."
                )
            if not self.drivable_data_path:
                raise ValueError(
                    "drivable_data_path is required when drivable loss is enabled."
                )
        if self.turn_route_loss_weight < 0:
            raise ValueError("turn_route_loss_weight must be non-negative.")
        if (
            self.turn_route_loss_start_step < 0
            or self.turn_route_loss_ramp_steps <= 0
        ):
            raise ValueError("Invalid turn-route loss ramp configuration.")
        if (
            not 0 <= self.turn_min_reference_angle <= math.pi
            or not 0 <= self.turn_tolerance <= math.pi
        ):
            raise ValueError(
                "Turn-route angle thresholds must lie in [0, pi]."
            )
        if not 0 <= self.turn_direction_consistency_min <= 1:
            raise ValueError(
                "turn_direction_consistency_min must lie in [0,1]."
            )
        if self.turn_risk_margin < 0:
            raise ValueError("turn_risk_margin must be non-negative.")
        if self.turn_risk_temperature <= 0:
            raise ValueError("turn_risk_temperature must be positive.")
        if self.turn_lateral_margin < 0:
            raise ValueError("turn_lateral_margin must be non-negative.")
        if not 0 < self.turn_time_weight_min <= 1:
            raise ValueError("turn_time_weight_min must lie in (0,1].")
        if (
            self.turn_component_weight < 0
            or self.turn_heading_weight < 0
            or self.turn_lateral_weight < 0
        ):
            raise ValueError("Turn-route component weights must be non-negative.")
        if (
            self.turn_route_loss_weight > 0
            and (
                self.turn_component_weight
                + self.turn_heading_weight
                + self.turn_lateral_weight
                <= 0
            )
        ):
            raise ValueError(
                "Positive turn-route loss requires at least one component."
            )
        if self.turn_route_loss_weight > 0:
            if not self.enabled:
                raise ValueError(
                    "A positive turn_route_loss_weight requires Level-k."
                )
            if not self.drivable_data_path or not self.progress_route_data_path:
                raise ValueError(
                    "Turn-route loss requires drivable and progress-route data."
                )
        if self.progress_loss_weight < 0:
            raise ValueError("progress_loss_weight must be non-negative.")
        if not 0 <= self.progress_tolerance_ratio < 1:
            raise ValueError("progress_tolerance_ratio must lie in [0, 1).")
        if self.progress_min_gt_distance < 0:
            raise ValueError("progress_min_gt_distance must be non-negative.")
        if self.progress_loss_start_step < 0:
            raise ValueError("progress_loss_start_step must be non-negative.")
        if self.progress_loss_ramp_steps <= 0:
            raise ValueError("progress_loss_ramp_steps must be positive.")
        if not 0.0 <= self.pdm_progress_tolerance_ratio < 1.0:
            raise ValueError("pdm_progress_tolerance_ratio must lie in [0,1).")
        if self.pdm_progress_distance_threshold < 0.0:
            raise ValueError("pdm_progress_distance_threshold must be non-negative.")
        if self.pdm_rollout_enabled and not self.enabled:
            raise ValueError("PDM rollout requires Level-k training.")
        if self.progress_safe_loss_weight < 0:
            raise ValueError("progress_safe_loss_weight must be non-negative.")
        if not 0 <= self.progress_route_min_coverage <= 1:
            raise ValueError("progress_route_min_coverage must lie in [0,1].")
        if self.progress_projection_temperature <= 0:
            raise ValueError("progress_projection_temperature must be positive.")
        if self.progress_safe_margin < 0 or self.progress_safe_min_gt_progress < 0:
            raise ValueError("Safe progress margins must be non-negative.")
        if (
            self.progress_drivable_gate_temperature <= 0
            or self.progress_ttc_gate_temperature <= 0
        ):
            raise ValueError("Safe progress gate temperatures must be positive.")
        if (
            self.progress_safe_loss_start_step < 0
            or self.progress_safe_loss_ramp_steps <= 0
        ):
            raise ValueError("Invalid safe progress loss ramp configuration.")
        if self.progress_route_data_path is not None and not isinstance(
            self.progress_route_data_path, str
        ):
            raise TypeError("progress_route_data_path must be a string or None.")
        if self.progress_safe_loss_weight > 0:
            if not self.enabled:
                raise ValueError(
                    "A positive progress_safe_loss_weight requires Level-k."
                )
            if not self.progress_route_data_path:
                raise ValueError(
                    "progress_route_data_path is required for safe progress loss."
                )
            if not self.drivable_data_path:
                raise ValueError(
                    "drivable_data_path is required for safe progress gating."
                )
        if self.retain_loss_weight < 0:
            raise ValueError("retain_loss_weight must be non-negative.")
        if not 0 <= self.retain_min_coverage <= 1:
            raise ValueError("retain_min_coverage must lie in [0,1].")
        if not 0 <= self.retain_min_safety_score < 1:
            raise ValueError("retain_min_safety_score must lie in [0,1).")
        if self.retain_heading_weight < 0:
            raise ValueError("retain_heading_weight must be non-negative.")
        if self.retain_loss_start_step < 0 or self.retain_loss_ramp_steps <= 0:
            raise ValueError("Invalid retain loss ramp configuration.")
        if self.retain_data_path is not None and not isinstance(
            self.retain_data_path, str
        ):
            raise TypeError("retain_data_path must be a string or None.")
        if self.retain_loss_weight > 0:
            if not self.enabled:
                raise ValueError("A positive retain_loss_weight requires Level-k.")
            if not self.retain_data_path:
                raise ValueError("retain_data_path is required for retain loss.")
        if self.safe_trajectory_loss_weight < 0:
            raise ValueError("safe_trajectory_loss_weight must be non-negative.")
        if self.safe_trajectory_heading_weight < 0:
            raise ValueError("safe_trajectory_heading_weight must be non-negative.")
        if self.safe_trajectory_loss_start_step < 0:
            raise ValueError(
                "safe_trajectory_loss_start_step must be non-negative."
            )
        if self.safe_trajectory_loss_ramp_steps <= 0:
            raise ValueError("safe_trajectory_loss_ramp_steps must be positive.")
        if self.hard_negative_loss_weight < 0:
            raise ValueError("hard_negative_loss_weight must be non-negative.")
        if not 0.0 <= self.hard_negative_sample_ratio < 1.0:
            raise ValueError(
                "hard_negative_sample_ratio must lie in [0, 1)."
            )
        if not 0.0 <= self.hard_negative_margin_ratio <= 1.0:
            raise ValueError("hard_negative_margin_ratio must lie in [0, 1].")
        if self.hard_negative_max_margin <= 0:
            raise ValueError("hard_negative_max_margin must be positive.")
        if self.hard_negative_loss_start_step < 0:
            raise ValueError(
                "hard_negative_loss_start_step must be non-negative."
            )
        if self.hard_negative_loss_ramp_steps <= 0:
            raise ValueError("hard_negative_loss_ramp_steps must be positive.")
        if not 0.0 <= self.hard_negative_min_coverage <= 1.0:
            raise ValueError(
                "hard_negative_min_coverage must lie in [0, 1]."
            )
        if (
            self.hard_negative_data_path is not None
            and not isinstance(self.hard_negative_data_path, str)
        ):
            raise TypeError(
                "hard_negative_data_path must be a string or None."
            )
        if (
            self.hard_negative_split_path is not None
            and not isinstance(self.hard_negative_split_path, str)
        ):
            raise TypeError(
                "hard_negative_split_path must be a string or None."
            )
        if self.hard_negative_enabled:
            if not self.enabled:
                raise ValueError(
                    "hard_negative_enabled requires Level-k to be enabled."
                )
            if not self.hard_negative_data_path:
                raise ValueError(
                    "hard_negative_data_path is required when hard-negative "
                    "supervision is enabled."
                )
            if self.hard_negative_loss_weight <= 0:
                raise ValueError(
                    "hard_negative_loss_weight must be positive when "
                    "hard-negative supervision is enabled."
                )
        elif self.hard_negative_sample_ratio > 0.0:
            raise ValueError(
                "hard_negative_sample_ratio requires hard-negative "
                "supervision to be enabled."
            )
        if not 0.0 <= self.safety_hard_sample_ratio < 1.0:
            raise ValueError("safety_hard_sample_ratio must lie in [0, 1).")
        if (
            self.safety_hard_data_path is not None
            and not isinstance(self.safety_hard_data_path, str)
        ):
            raise TypeError("safety_hard_data_path must be a string or None.")
        if self.safety_hard_enabled:
            if not self.enabled:
                raise ValueError("safety_hard_enabled requires Level-k.")
            if not self.safety_hard_data_path:
                raise ValueError(
                    "safety_hard_data_path is required when safety-hard "
                    "sampling is enabled."
                )
            if not 0.0 < self.safety_hard_sample_ratio < 1.0:
                raise ValueError(
                    "Enabled safety-hard sampling requires a ratio in (0, 1)."
                )
        elif self.safety_hard_sample_ratio > 0.0:
            raise ValueError(
                "safety_hard_sample_ratio requires safety-hard sampling."
            )
        if (
            self.ttc_swept_box_loss_weight > 0
            or self.progress_safe_loss_weight > 0
            or self.turn_route_loss_weight > 0
            or self.retain_loss_weight > 0
        ) and not self.hard_negative_split_path:
            raise ValueError(
                "hard_negative_split_path is required to protect fixed-dev "
                "samples when new auxiliary losses are enabled."
            )
        if self.ego_vehicle_length <= 0:
            raise ValueError("ego_vehicle_length must be positive.")
        if self.ego_vehicle_width <= 0:
            raise ValueError("ego_vehicle_width must be positive.")
        if self.ego_rear_axle_to_center < 0:
            raise ValueError("ego_rear_axle_to_center must be non-negative.")
        if self.ego_action_lr <= 0:
            raise ValueError("ego_action_lr must be positive.")
        if self.other_agent_lr <= 0:
            raise ValueError("other_agent_lr must be positive.")
        if self.strategy_lr <= 0:
            raise ValueError("strategy_lr must be positive.")
        if (
            self.validation_report_path is not None
            and not isinstance(self.validation_report_path, str)
        ):
            raise TypeError("validation_report_path must be a string or None.")
    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> "LevelKConfig":
        """Build a validated config; missing sections keep Level-k disabled."""

        if values is None:
            return cls()
        return cls(**dict(values))


def synchronize_level_k_model_config(
    config: LevelKConfig,
    model_config: MutableMapping[str, Any],
) -> None:
    """Enable and align optional Transformer modules from one Level-k switch.

    With Level-k disabled this function is an exact no-op, preserving the
    original DriveLaW model construction and checkpoint behavior.
    """

    if not isinstance(model_config, MutableMapping):
        raise TypeError("diffusion_model.config must be a mutable mapping.")
    if not config.enabled:
        return
    if model_config.get("action_expert") is not True:
        raise ValueError("Level-k requires diffusion_model.config.action_expert=true.")

    model_config.update(
        {
            "action_strategy_conditioning": True,
            "other_agent_expert": True,
            "other_agent_horizon_steps": config.horizon_steps,
            "other_agent_position_scale": (
                config.other_trajectory_position_scale
            ),
            "other_agent_velocity_scale": config.strategy_velocity_scale,
            "other_agent_size_scale": config.strategy_size_scale,
            "strategy_max_reasoning_level": config.max_reasoning_level,
            "strategy_interval_length": config.interval_length,
        }
    )


def validate_level_k_trajectory_sampling(
    config: LevelKConfig,
    *,
    num_poses: int,
    interval_length: float,
) -> None:
    """Ensure Level-k strategies and NAVSIM supervision share one time grid."""

    if not config.enabled:
        return
    if config.horizon_steps != int(num_poses):
        raise ValueError(
            "level_k.horizon_steps must equal "
            f"trajectory_sampling.num_poses ({num_poses})."
        )
    if not math.isclose(
        config.interval_length,
        float(interval_length),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "level_k.interval_length must equal "
            "trajectory_sampling.interval_length "
            f"({interval_length})."
        )
