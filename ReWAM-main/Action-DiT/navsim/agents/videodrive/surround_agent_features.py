"""Causal surrounding-agent features for Level-K inference."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from navsim.agents.videodrive.level_k.geometry import normalize_angle
from navsim.agents.videodrive.level_k.level_zero_map import (
    MapLevelZeroReferenceBuilder,
)
from navsim.agents.videodrive.level_k.oracle_data import (
    validate_oracle_agent_inputs,
)
from navsim.agents.videodrive.level_k.types import LevelZeroReferenceBatch
from navsim.common.enums import BoundingBoxIndex


_VEHICLE_NAME = "vehicle"
_VEHICLE_TYPE_ID = 0
_STATE_DIM = 8


def _valid_box(box: np.ndarray) -> bool:
    return (
        box.shape[0] > int(BoundingBoxIndex.HEADING)
        and np.isfinite(box[: int(BoundingBoxIndex.HEADING) + 1]).all()
        and float(box[BoundingBoxIndex.LENGTH]) > 0.0
        and float(box[BoundingBoxIndex.WIDTH]) > 0.0
    )


def _valid_velocity(velocity: np.ndarray) -> bool:
    return (
        velocity.ndim == 1
        and velocity.shape[0] >= 2
        and np.isfinite(velocity[:2]).all()
    )


class SurroundAgentFeatureBuilder:
    """Build current-frame surrounding-agent inputs without future labels."""

    def __init__(
        self,
        *,
        max_agents: int = 10,
        agent_range: float = 80.0,
        horizon_steps: int = 8,
        interval_length: float = 0.5,
        level_zero_mode: str = "constant_heading",
        level_zero_lane_search_radius: float = 8.0,
        level_zero_max_lateral_error: float = 4.0,
        level_zero_max_heading_error: float = math.pi / 3.0,
        level_zero_route_search_depth: int = 30,
    ) -> None:
        if isinstance(max_agents, bool) or not isinstance(max_agents, Integral):
            raise TypeError("max_agents must use an integer value.")
        if isinstance(horizon_steps, bool) or not isinstance(
            horizon_steps, Integral
        ):
            raise TypeError("horizon_steps must use an integer value.")
        if isinstance(agent_range, bool) or not isinstance(agent_range, Real):
            raise TypeError("agent_range must use a real numeric value.")
        if not math.isfinite(float(agent_range)):
            raise ValueError("agent_range must be finite.")
        if max_agents <= 0 or agent_range <= 0 or horizon_steps <= 0:
            raise ValueError(
                "max_agents, agent_range, and horizon_steps must be positive."
            )
        if level_zero_mode not in {"constant_heading", "map_lane_following"}:
            raise ValueError(
                "level_zero_mode must be 'constant_heading' or "
                "'map_lane_following'."
            )

        self.max_agents = int(max_agents)
        self.agent_range = float(agent_range)
        self.horizon_steps = int(horizon_steps)
        self.level_zero_mode = level_zero_mode
        self._level_zero_map_builder = None
        if level_zero_mode == "map_lane_following":
            self._level_zero_map_builder = MapLevelZeroReferenceBuilder(
                horizon_steps=self.horizon_steps,
                interval_length=interval_length,
                lane_search_radius=level_zero_lane_search_radius,
                max_lateral_error=level_zero_max_lateral_error,
                max_heading_error=level_zero_max_heading_error,
                route_search_depth=level_zero_route_search_depth,
            )

    def _select_current_agents(
        self, annotations: Any
    ) -> List[Tuple[float, str, int]]:
        selected: List[Tuple[float, str, int]] = []
        for index, (name, token) in enumerate(
            zip(annotations.names, annotations.track_tokens)
        ):
            if str(name) != _VEHICLE_NAME:
                continue
            box = np.asarray(annotations.boxes[index], dtype=np.float64)
            velocity = np.asarray(
                annotations.velocity_3d[index], dtype=np.float64
            )
            if not _valid_box(box) or not _valid_velocity(velocity):
                continue
            distance = float(np.linalg.norm(box[BoundingBoxIndex.POINT2D]))
            token_string = str(token)
            if distance <= self.agent_range and token_string:
                selected.append((distance, token_string, index))

        selected.sort(key=lambda item: (item[0], item[1]))
        unique_selected: List[Tuple[float, str, int]] = []
        seen_tokens: set[str] = set()
        for candidate in selected:
            if candidate[1] in seen_tokens:
                continue
            seen_tokens.add(candidate[1])
            unique_selected.append(candidate)
            if len(unique_selected) == self.max_agents:
                break
        return unique_selected

    @staticmethod
    def _current_frame(scene: Any) -> Any:
        current_index = scene.scene_metadata.num_history_frames - 1
        if not 0 <= current_index < len(scene.frames):
            raise ValueError("Scene does not contain the configured current frame.")
        return scene.frames[current_index]

    def _build_current_inputs(
        self, annotations: Any
    ) -> Dict[str, np.ndarray]:
        current_states = np.zeros((self.max_agents, _STATE_DIM), dtype=np.float32)
        agent_types = np.full((self.max_agents,), -1, dtype=np.int64)
        agent_mask = np.zeros((self.max_agents,), dtype=np.bool_)

        for slot, (_, _, annotation_index) in enumerate(
            self._select_current_agents(annotations)
        ):
            box = np.asarray(annotations.boxes[annotation_index], dtype=np.float64)
            heading = normalize_angle(float(box[BoundingBoxIndex.HEADING]))
            velocity = np.asarray(
                annotations.velocity_3d[annotation_index], dtype=np.float64
            )[:2]
            current_states[slot] = np.asarray(
                [
                    box[BoundingBoxIndex.X],
                    box[BoundingBoxIndex.Y],
                    np.sin(heading),
                    np.cos(heading),
                    velocity[0],
                    velocity[1],
                    box[BoundingBoxIndex.LENGTH],
                    box[BoundingBoxIndex.WIDTH],
                ],
                dtype=np.float32,
            )
            agent_types[slot] = _VEHICLE_TYPE_ID
            agent_mask[slot] = True

        return {
            "agent_current_states": current_states,
            "agent_types": agent_types,
            "agent_mask": agent_mask,
        }

    def compute_current_inputs(self, scene: Any) -> Dict[str, torch.Tensor]:
        """Return current annotation states without reading future frames."""

        current_frame = self._current_frame(scene)
        arrays = self._build_current_inputs(current_frame.annotations)
        result = {name: torch.from_numpy(value) for name, value in arrays.items()}
        validate_oracle_agent_inputs(result)
        result.update(self._compute_level_zero_references(scene, arrays))
        return result

    def _compute_level_zero_references(
        self,
        scene: Any,
        current_inputs: Dict[str, np.ndarray],
    ) -> Dict[str, torch.Tensor]:
        if self._level_zero_map_builder is None:
            return {}
        arrays = self._level_zero_map_builder.build(
            scene=scene,
            current_states=current_inputs["agent_current_states"],
            agent_mask=current_inputs["agent_mask"],
        )
        result = {
            name: torch.from_numpy(np.asarray(value))
            for name, value in arrays.items()
        }
        references = LevelZeroReferenceBatch.from_dict(result, add_batch_dim=True)
        references.validate(
            agent_mask=torch.from_numpy(
                current_inputs["agent_mask"]
            ).unsqueeze(0),
            horizon_steps=self.horizon_steps,
        )
        return result

    def selected_track_tokens(self, scene: Any) -> Tuple[str, ...]:
        """Return track tokens in padded-slot selection order."""

        current_frame = self._current_frame(scene)
        return tuple(
            track_token
            for _, track_token, _ in self._select_current_agents(
                current_frame.annotations
            )
        )


__all__ = ["SurroundAgentFeatureBuilder"]
