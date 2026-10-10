"""Causal map-lane references for deterministic Level-0 strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Iterable, Sequence

import numpy as np
import numpy.typing as npt

from .geometry import global_pose_to_local, local_pose_to_global, normalize_angle


PathArray = npt.NDArray[np.float64]


def _finite_pose_array(value: Any, *, name: str) -> PathArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite SE(2) pose.")
    return array


def _discrete_path(edge: Any) -> PathArray:
    """Return one lane baseline as finite ``[P, 3]`` world poses."""

    states = getattr(getattr(edge, "baseline_path", None), "discrete_path", None)
    if states is None:
        return np.zeros((0, 3), dtype=np.float64)
    poses = []
    for state in states:
        if hasattr(state, "array"):
            pose = np.asarray(state.array, dtype=np.float64)
        else:
            pose = np.asarray(
                [state.x, state.y, state.heading],
                dtype=np.float64,
            )
        if pose.shape == (3,) and np.isfinite(pose).all():
            poses.append(pose)
    if not poses:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(poses)


def _append_path(base: list[PathArray], segment: PathArray) -> None:
    if segment.shape[0] == 0:
        return
    if base and np.linalg.norm(base[-1][:2] - segment[0, :2]) <= 1e-6:
        segment = segment[1:]
    base.extend(segment)


def _path_length(path: PathArray) -> float:
    if path.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum())


def _edge_entry_heading(edge: Any) -> float | None:
    path = _discrete_path(edge)
    if path.shape[0] < 2:
        return None
    delta = path[1, :2] - path[0, :2]
    if np.linalg.norm(delta) <= 1e-9:
        return float(path[0, 2])
    return float(math.atan2(delta[1], delta[0]))


def _edge_exit_heading(edge: Any) -> float | None:
    path = _discrete_path(edge)
    if path.shape[0] < 2:
        return None
    delta = path[-1, :2] - path[-2, :2]
    if np.linalg.norm(delta) <= 1e-9:
        return float(path[-1, 2])
    return float(math.atan2(delta[1], delta[0]))


def select_minimum_heading_successor(
    edge: Any,
    *,
    allowed_edge_ids: set[str] | None = None,
    visited_edge_ids: set[str] | None = None,
) -> Any | None:
    """Select the valid outgoing edge with the smallest heading change."""

    outgoing = list(getattr(edge, "outgoing_edges", ()) or ())
    previous_heading = _edge_exit_heading(edge)
    if previous_heading is None:
        return None
    visited = visited_edge_ids or set()
    scored = []
    for candidate in outgoing:
        candidate_id = str(getattr(candidate, "id", ""))
        if not candidate_id or candidate_id in visited:
            continue
        if allowed_edge_ids is not None and candidate_id not in allowed_edge_ids:
            continue
        entry_heading = _edge_entry_heading(candidate)
        if entry_heading is None:
            continue
        heading_error = abs(normalize_angle(entry_heading - previous_heading))
        scored.append((heading_error, candidate_id, candidate))
    return min(scored, default=(None, None, None))[-1]


def extend_lane_centerline(
    start_edge: Any,
    *,
    required_length: float,
    max_depth: int,
    allowed_edge_ids: set[str] | None = None,
) -> PathArray:
    """Greedily extend a lane path until it covers ``required_length``."""

    if required_length < 0 or not math.isfinite(required_length):
        raise ValueError("required_length must be finite and non-negative.")
    if max_depth <= 0:
        raise ValueError("max_depth must be positive.")
    edge = start_edge
    edge_id = str(getattr(edge, "id", ""))
    visited = {edge_id} if edge_id else set()
    points: list[PathArray] = []
    for _ in range(max_depth):
        _append_path(points, _discrete_path(edge))
        path = np.asarray(points, dtype=np.float64)
        if _path_length(path) >= required_length:
            return path
        next_edge = select_minimum_heading_successor(
            edge,
            allowed_edge_ids=allowed_edge_ids,
            visited_edge_ids=visited,
        )
        if next_edge is None:
            break
        edge = next_edge
        next_id = str(getattr(edge, "id", ""))
        if next_id:
            visited.add(next_id)
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def project_and_sample_centerline(
    path: PathArray,
    *,
    start_xy: npt.ArrayLike,
    travel_distances: npt.ArrayLike,
) -> PathArray | None:
    """Project onto a polyline and sample complete future world poses."""

    path = np.asarray(path, dtype=np.float64)
    start = np.asarray(start_xy, dtype=np.float64)
    distances = np.asarray(travel_distances, dtype=np.float64)
    if (
        path.ndim != 2
        or path.shape[1] != 3
        or path.shape[0] < 2
        or start.shape != (2,)
        or distances.ndim != 1
        or not np.isfinite(path).all()
        or not np.isfinite(start).all()
        or not np.isfinite(distances).all()
        or np.any(distances < 0)
    ):
        return None

    starts = path[:-1, :2]
    deltas = path[1:, :2] - starts
    lengths = np.linalg.norm(deltas, axis=1)
    valid = lengths > 1e-9
    if not np.any(valid):
        return None
    starts = starts[valid]
    deltas = deltas[valid]
    lengths = lengths[valid]
    squared_lengths = lengths**2
    fractions = np.clip(
        np.sum((start[None] - starts) * deltas, axis=1) / squared_lengths,
        0.0,
        1.0,
    )
    projections = starts + fractions[:, None] * deltas
    nearest = int(np.argmin(np.linalg.norm(projections - start[None], axis=1)))

    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    start_s = cumulative[nearest] + fractions[nearest] * lengths[nearest]
    query_s = start_s + distances
    if query_s.size and query_s[-1] > cumulative[-1] + 1e-6:
        return None

    sampled = np.zeros((distances.size, 3), dtype=np.float64)
    for index, query in enumerate(query_s):
        segment = min(int(np.searchsorted(cumulative, query, side="right") - 1), len(lengths) - 1)
        segment = max(segment, 0)
        fraction = (query - cumulative[segment]) / lengths[segment]
        sampled[index, :2] = starts[segment] + fraction * deltas[segment]
        sampled[index, 2] = math.atan2(deltas[segment, 1], deltas[segment, 0])
    return sampled


def lane_match_score(
    edge: Any,
    *,
    pose: npt.ArrayLike,
    max_lateral_error: float,
    max_heading_error: float,
) -> tuple[float, float, str] | None:
    """Score a candidate lane by lateral distance, then heading error."""

    pose_array = _finite_pose_array(pose, name="pose")
    path = _discrete_path(edge)
    if path.shape[0] < 2:
        return None
    sampled = project_and_sample_centerline(
        path,
        start_xy=pose_array[:2],
        travel_distances=np.asarray([0.0]),
    )
    if sampled is None:
        return None
    lateral_error = float(np.linalg.norm(sampled[0, :2] - pose_array[:2]))
    heading_error = abs(normalize_angle(float(sampled[0, 2] - pose_array[2])))
    if lateral_error > max_lateral_error or heading_error > max_heading_error:
        return None
    return lateral_error, heading_error, str(getattr(edge, "id", ""))


def select_best_lane(
    candidates: Iterable[Any],
    *,
    pose: npt.ArrayLike,
    max_lateral_error: float,
    max_heading_error: float,
) -> Any | None:
    scored = []
    for edge in candidates:
        score = lane_match_score(
            edge,
            pose=pose,
            max_lateral_error=max_lateral_error,
            max_heading_error=max_heading_error,
        )
        if score is not None:
            scored.append((*score, edge))
    return min(scored, default=(None, None, None, None))[-1]


@dataclass(frozen=True)
class MapLevelZeroReferenceBuilder:
    """Build map-constrained Ego and Other references from one current Scene."""

    horizon_steps: int = 8
    interval_length: float = 0.5
    lane_search_radius: float = 8.0
    max_lateral_error: float = 4.0
    max_heading_error: float = math.pi / 3.0
    route_search_depth: int = 30

    def __post_init__(self) -> None:
        if isinstance(self.horizon_steps, bool) or not isinstance(self.horizon_steps, Integral):
            raise TypeError("horizon_steps must use an integer value.")
        if isinstance(self.route_search_depth, bool) or not isinstance(self.route_search_depth, Integral):
            raise TypeError("route_search_depth must use an integer value.")
        for name in (
            "interval_length",
            "lane_search_radius",
            "max_lateral_error",
            "max_heading_error",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must use a real numeric value.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if self.horizon_steps <= 0 or self.route_search_depth <= 0:
            raise ValueError("Step counts and search depth must be positive.")
        if self.interval_length <= 0 or self.lane_search_radius <= 0:
            raise ValueError("Time interval and search radius must be positive.")
        if self.max_lateral_error <= 0 or not 0 < self.max_heading_error <= math.pi:
            raise ValueError("Lane-matching thresholds are invalid.")

    def _route_lanes(self, scene: Any, current_frame: Any) -> list[Any]:
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer

        lanes = []
        for roadblock_id in dict.fromkeys(current_frame.roadblock_ids):
            block = scene.map_api.get_map_object(
                roadblock_id,
                SemanticMapLayer.ROADBLOCK,
            )
            if block is None:
                block = scene.map_api.get_map_object(
                    roadblock_id,
                    SemanticMapLayer.ROADBLOCK_CONNECTOR,
                )
            if block is not None:
                lanes.extend(list(getattr(block, "interior_edges", ()) or ()))
        return lanes

    def _nearby_lanes(self, scene: Any, global_pose: PathArray) -> list[Any]:
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer

        point = Point2D(float(global_pose[0]), float(global_pose[1]))
        layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        objects = scene.map_api.get_proximal_map_objects(
            point=point,
            radius=self.lane_search_radius,
            layers=layers,
        )
        return [edge for layer in layers for edge in objects.get(layer, ())]

    def _sample_reference(
        self,
        *,
        edge: Any,
        global_pose: PathArray,
        speed: float,
        anchor_ego_pose: PathArray,
        relative_origin_xy: PathArray,
        allowed_edge_ids: set[str] | None,
    ) -> PathArray | None:
        times = np.arange(1, self.horizon_steps + 1, dtype=np.float64) * self.interval_length
        distances = max(float(speed), 0.0) * times
        initial_path = _discrete_path(edge)
        projection = project_and_sample_centerline(
            initial_path,
            start_xy=global_pose[:2],
            travel_distances=np.asarray([0.0]),
        )
        if projection is None:
            return None
        # Include the distance from the lane start to the current projection
        # when deciding how much successor centerline must be collected.
        required_length = _path_length(initial_path) + float(distances[-1]) + 1.0
        path = extend_lane_centerline(
            edge,
            required_length=required_length,
            max_depth=self.route_search_depth,
            allowed_edge_ids=allowed_edge_ids,
        )
        sampled_global = project_and_sample_centerline(
            path,
            start_xy=global_pose[:2],
            travel_distances=distances,
        )
        if sampled_global is None:
            return None
        local = np.stack(
            [global_pose_to_local(pose, anchor_ego_pose) for pose in sampled_global]
        )
        local[:, :2] -= relative_origin_xy[None]
        return np.stack(
            (local[:, 0], local[:, 1], np.sin(local[:, 2]), np.cos(local[:, 2])),
            axis=-1,
        ).astype(np.float32)

    def build(
        self,
        *,
        scene: Any,
        current_states: npt.ArrayLike,
        agent_mask: npt.ArrayLike,
    ) -> dict[str, npt.NDArray[Any]]:
        """Build references without reading any frame after current time."""

        states = np.asarray(current_states, dtype=np.float64)
        mask = np.asarray(agent_mask, dtype=np.bool_)
        if states.ndim != 2 or states.shape[-1] != 8:
            raise ValueError("current_states must have shape [N, 8].")
        if mask.shape != states.shape[:1]:
            raise ValueError("agent_mask must have shape [N].")

        current_index = scene.scene_metadata.num_history_frames - 1
        current_frame = scene.frames[current_index]
        anchor = _finite_pose_array(
            current_frame.ego_status.ego_pose,
            name="current Ego pose",
        )
        ego_reference = np.zeros((self.horizon_steps, 4), dtype=np.float32)
        other_references = np.zeros(
            (states.shape[0], self.horizon_steps, 4),
            dtype=np.float32,
        )
        ego_valid = np.asarray(False, dtype=np.bool_)
        other_valid = np.zeros(states.shape[0], dtype=np.bool_)

        route_lanes = self._route_lanes(scene, current_frame)
        ego_edge = select_best_lane(
            route_lanes,
            pose=anchor,
            max_lateral_error=self.max_lateral_error,
            max_heading_error=self.max_heading_error,
        )
        if ego_edge is not None:
            velocity = np.asarray(current_frame.ego_status.ego_velocity, dtype=np.float64)
            ego_speed = float(velocity[0]) if velocity.size >= 1 and np.isfinite(velocity[0]) else 0.0
            sampled = self._sample_reference(
                edge=ego_edge,
                global_pose=anchor,
                speed=ego_speed,
                anchor_ego_pose=anchor,
                relative_origin_xy=np.zeros(2, dtype=np.float64),
                allowed_edge_ids={str(edge.id) for edge in route_lanes},
            )
            if sampled is not None:
                ego_reference = sampled
                ego_valid[...] = True

        for slot in np.flatnonzero(mask):
            heading = math.atan2(states[slot, 2], states[slot, 3])
            local_pose = np.asarray(
                [states[slot, 0], states[slot, 1], heading],
                dtype=np.float64,
            )
            global_pose = local_pose_to_global(local_pose, anchor)
            candidates = self._nearby_lanes(scene, global_pose)
            edge = select_best_lane(
                candidates,
                pose=global_pose,
                max_lateral_error=self.max_lateral_error,
                max_heading_error=self.max_heading_error,
            )
            if edge is None:
                continue
            longitudinal_speed = max(
                float(
                    states[slot, 4] * math.cos(heading)
                    + states[slot, 5] * math.sin(heading)
                ),
                0.0,
            )
            sampled = self._sample_reference(
                edge=edge,
                global_pose=global_pose,
                speed=longitudinal_speed,
                anchor_ego_pose=anchor,
                relative_origin_xy=states[slot, :2],
                allowed_edge_ids=None,
            )
            if sampled is not None:
                other_references[slot] = sampled
                other_valid[slot] = True

        return {
            "level_zero_ego_reference": ego_reference,
            "level_zero_ego_reference_mask": ego_valid,
            "level_zero_other_references": other_references,
            "level_zero_other_reference_mask": other_valid,
        }
