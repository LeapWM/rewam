#!/usr/bin/env python3
"""Model-independent interaction intensity metrics for NAVSIM metric caches."""

from __future__ import annotations

import json
import lzma
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from nuplan.common.actor_state.state_representation import TimePoint


INTERACTION_DEFINITION_VERSION = 2


@dataclass(frozen=True)
class InteractionThresholds:
    """Physical thresholds and normalizers used by the v2 score."""

    horizon_s: float = 4.0
    sample_interval_s: float = 0.5
    nearby_radius_m: float = 35.0
    min_agent_displacement_m: float = 1.0
    min_agent_peak_speed_mps: float = 0.5
    max_path_conflict_distance_m: float = 3.0
    path_conflict_tolerance_m: float = 0.25
    max_conflict_time_gap_s: float = 1.5
    max_synchronized_distance_m: float = 3.0
    proximity_scale_m: float = 3.0
    path_scale_m: float = 3.0
    temporal_scale_s: float = 1.5
    closing_speed_scale_mps: float = 5.0
    agent_speed_scale_mps: float = 8.0
    reference_speed_drop_scale_mps: float = 2.0

    def validate(self) -> None:
        values = asdict(self)
        non_positive = [
            key
            for key, value in values.items()
            if not math.isfinite(float(value)) or float(value) <= 0.0
        ]
        if non_positive:
            raise ValueError(
                f"Interaction thresholds must be finite and positive: {non_positive}"
            )


@dataclass(frozen=True)
class AgentInteraction:
    track_token: str
    tracked_object_type: str
    current_distance_m: float
    displacement_m: float
    peak_speed_mps: float
    max_closing_speed_mps: float
    min_path_distance_m: float
    conflict_time_gap_s: float
    min_synchronized_distance_m: float
    first_close_approach_s: float
    proximity_score: float
    spatiotemporal_conflict_score: float
    closing_score: float
    motion_score: float
    interaction_intensity_v2: float
    moving: bool
    nearby: bool
    synchronized_close: bool
    path_time_conflict: bool
    eligible: bool


def _decay_score(value: float, scale: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(math.exp(-max(value, 0.0) / scale))


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def analyze_agent_geometry(
    *,
    track_token: str,
    tracked_object_type: str,
    times_s: np.ndarray,
    ego_polygons: Sequence[Any],
    agent_polygons: Sequence[Any | None],
    thresholds: InteractionThresholds,
) -> AgentInteraction:
    """Measure one dynamic agent against a fixed reference ego rollout."""
    available = [
        (index, polygon)
        for index, polygon in enumerate(agent_polygons)
        if polygon is not None
    ]
    if not available:
        raise ValueError("Agent has no polygons in the analysis horizon")

    first_index, first_polygon = available[0]
    current_distance = (
        float(ego_polygons[0].distance(first_polygon)) if first_index == 0 else math.inf
    )
    centers = np.asarray(
        [[polygon.centroid.x, polygon.centroid.y] for _, polygon in available],
        dtype=np.float64,
    )
    agent_times = np.asarray(
        [times_s[index] for index, _ in available], dtype=np.float64
    )
    displacement = float(np.linalg.norm(centers - centers[0], axis=1).max())
    if len(centers) > 1:
        delta_t = np.maximum(np.diff(agent_times), 1.0e-6)
        step_distance = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        peak_speed = float(np.max(step_distance / delta_t))
    else:
        peak_speed = 0.0

    synchronized_times = np.asarray(
        [times_s[index] for index, _ in available], dtype=np.float64
    )
    synchronized_distances = np.asarray(
        [float(ego_polygons[index].distance(polygon)) for index, polygon in available],
        dtype=np.float64,
    )
    min_synchronized = float(synchronized_distances.min())
    close_mask = synchronized_distances <= thresholds.max_synchronized_distance_m
    first_close = (
        float(synchronized_times[np.flatnonzero(close_mask)[0]])
        if close_mask.any()
        else math.inf
    )
    if len(synchronized_distances) > 1:
        synchronized_delta_t = np.maximum(np.diff(synchronized_times), 1.0e-6)
        closing_speeds = -np.diff(synchronized_distances) / synchronized_delta_t
        max_closing_speed = float(max(0.0, float(closing_speeds.max())))
    else:
        max_closing_speed = 0.0

    path_distances: list[tuple[int, int, float]] = []
    for ego_index, ego_polygon in enumerate(ego_polygons):
        for agent_index, agent_polygon in available:
            distance = float(ego_polygon.distance(agent_polygon))
            path_distances.append((ego_index, agent_index, distance))
    min_path_distance = min(distance for _, _, distance in path_distances)
    conflict_time_gap = math.inf
    if min_path_distance <= thresholds.max_path_conflict_distance_m:
        conflict_region_limit = min(
            thresholds.max_path_conflict_distance_m,
            min_path_distance + thresholds.path_conflict_tolerance_m,
        )
        conflict_time_gap = min(
            abs(float(times_s[ego_index] - times_s[agent_index]))
            for ego_index, agent_index, distance in path_distances
            if distance <= conflict_region_limit
        )

    moving = (
        displacement >= thresholds.min_agent_displacement_m
        or peak_speed >= thresholds.min_agent_peak_speed_mps
    )
    nearby = current_distance <= thresholds.nearby_radius_m
    synchronized_close = min_synchronized <= thresholds.max_synchronized_distance_m
    path_time_conflict = (
        min_path_distance <= thresholds.max_path_conflict_distance_m
        and conflict_time_gap <= thresholds.max_conflict_time_gap_s
    )
    eligible = moving and nearby and (synchronized_close or path_time_conflict)

    proximity_score = _decay_score(min_synchronized, thresholds.proximity_scale_m)
    spatial_score = _decay_score(min_path_distance, thresholds.path_scale_m)
    temporal_score = _decay_score(conflict_time_gap, thresholds.temporal_scale_s)
    spatiotemporal_conflict_score = spatial_score * temporal_score
    closing_score = float(
        np.clip(
            max_closing_speed / thresholds.closing_speed_scale_mps,
            0.0,
            1.0,
        )
    )
    motion_score = float(
        np.clip(peak_speed / thresholds.agent_speed_scale_mps, 0.0, 1.0)
    )
    intensity = float(
        0.35 * proximity_score
        + 0.35 * spatiotemporal_conflict_score
        + 0.20 * closing_score
        + 0.10 * motion_score
    )

    return AgentInteraction(
        track_token=track_token,
        tracked_object_type=tracked_object_type,
        current_distance_m=current_distance,
        displacement_m=displacement,
        peak_speed_mps=peak_speed,
        max_closing_speed_mps=max_closing_speed,
        min_path_distance_m=min_path_distance,
        conflict_time_gap_s=conflict_time_gap,
        min_synchronized_distance_m=min_synchronized,
        first_close_approach_s=first_close,
        proximity_score=proximity_score,
        spatiotemporal_conflict_score=spatiotemporal_conflict_score,
        closing_score=closing_score,
        motion_score=motion_score,
        interaction_intensity_v2=intensity,
        moving=moving,
        nearby=nearby,
        synchronized_close=synchronized_close,
        path_time_conflict=path_time_conflict,
        eligible=eligible,
    )


def aggregate_scene_interaction(
    agents: Sequence[AgentInteraction],
    *,
    reference_initial_speed_mps: float,
    reference_min_speed_mps: float,
    thresholds: InteractionThresholds,
) -> dict[str, object]:
    """Aggregate per-agent measurements without using model predictions."""
    eligible = sorted(
        (agent for agent in agents if agent.eligible),
        key=lambda agent: (-agent.interaction_intensity_v2, agent.track_token),
    )
    best = eligible[0] if eligible else None
    multiplicity_bonus = 0.05 * min(max(len(eligible) - 1, 0), 2)
    intensity = float(
        np.clip(
            (best.interaction_intensity_v2 if best is not None else 0.0)
            + multiplicity_bonus,
            0.0,
            1.0,
        )
    )
    speed_drop = max(0.0, reference_initial_speed_mps - reference_min_speed_mps)
    required_response = float(
        np.clip(
            speed_drop / thresholds.reference_speed_drop_scale_mps,
            0.0,
            1.0,
        )
    )

    details: list[dict[str, object]] = []
    for agent in eligible:
        payload = asdict(agent)
        for key in (
            "current_distance_m",
            "conflict_time_gap_s",
            "first_close_approach_s",
        ):
            payload[key] = _finite_or_none(float(payload[key]))
        details.append(payload)

    return {
        "interaction_definition_version": INTERACTION_DEFINITION_VERSION,
        "vehicle_count": len(agents),
        "nearby_vehicle_count": sum(agent.nearby for agent in agents),
        "moving_nearby_vehicle_count": sum(
            agent.nearby and agent.moving for agent in agents
        ),
        "interacting_vehicle_count": len(eligible),
        "best_interacting_agent_token": best.track_token if best else "",
        "best_agent_current_distance_m": (
            _finite_or_none(best.current_distance_m) if best else None
        ),
        "best_agent_displacement_m": best.displacement_m if best else None,
        "best_agent_peak_speed_mps": best.peak_speed_mps if best else None,
        "max_closing_speed_mps": best.max_closing_speed_mps if best else None,
        "min_path_distance_m": best.min_path_distance_m if best else None,
        "conflict_time_gap_s": (
            _finite_or_none(best.conflict_time_gap_s) if best else None
        ),
        "min_synchronized_distance_m": (
            best.min_synchronized_distance_m if best else None
        ),
        "first_close_approach_s": (
            _finite_or_none(best.first_close_approach_s) if best else None
        ),
        "synchronized_proximity_score": (best.proximity_score if best else 0.0),
        "spatiotemporal_conflict_score": (
            best.spatiotemporal_conflict_score if best else 0.0
        ),
        "closing_score": best.closing_score if best else 0.0,
        "agent_motion_score": best.motion_score if best else 0.0,
        "interaction_multiplicity_bonus": multiplicity_bonus,
        "interaction_intensity_v2": intensity,
        "reference_initial_speed_mps": reference_initial_speed_mps,
        "reference_min_speed_mps": reference_min_speed_mps,
        "reference_speed_drop_mps": speed_drop,
        "required_response_score": required_response,
        "reference_stops": bool(
            reference_initial_speed_mps > 1.0 and reference_min_speed_mps < 0.5
        ),
        "interacting_agents_json": json.dumps(details, sort_keys=True),
    }


def analyze_metric_cache(
    token: str,
    cache_path: Path,
    thresholds: InteractionThresholds,
) -> dict[str, object]:
    """Load one metric cache and compute model-independent scene intensity."""
    with lzma.open(cache_path, "rb") as stream:
        cache = pickle.load(stream)

    available_horizon = (
        cache.trajectory.end_time.time_s - cache.trajectory.start_time.time_s
    )
    horizon = min(thresholds.horizon_s, available_horizon)
    times_s = np.arange(
        0.0,
        horizon + thresholds.sample_interval_s / 2.0,
        thresholds.sample_interval_s,
        dtype=np.float64,
    )
    time_points = [
        TimePoint(cache.trajectory.start_time.time_us + int(round(value * 1.0e6)))
        for value in times_s
    ]
    ego_states = cache.trajectory.get_state_at_times(time_points)
    ego_polygons = [state.car_footprint.geometry for state in ego_states]
    reference_speeds = np.asarray(
        [state.dynamic_car_state.speed for state in ego_states],
        dtype=np.float64,
    )

    observation = cache.observation
    observation_interval = float(observation._sample_interval)
    observation_indices = [
        int(round(value / observation_interval)) for value in times_s
    ]
    agents: list[AgentInteraction] = []
    for track_token, tracked_object in observation.unique_objects.items():
        type_name = tracked_object.tracked_object_type.name.lower()
        if type_name != "vehicle":
            continue
        polygons = []
        for observation_index in observation_indices:
            occupancy_map = observation[observation_index]
            polygons.append(
                occupancy_map[track_token]
                if track_token in occupancy_map.token_to_idx
                else None
            )
        try:
            agents.append(
                analyze_agent_geometry(
                    track_token=track_token,
                    tracked_object_type=type_name,
                    times_s=times_s,
                    ego_polygons=ego_polygons,
                    agent_polygons=polygons,
                    thresholds=thresholds,
                )
            )
        except ValueError:
            continue

    result = aggregate_scene_interaction(
        agents,
        reference_initial_speed_mps=float(reference_speeds[0]),
        reference_min_speed_mps=float(reference_speeds.min()),
        thresholds=thresholds,
    )
    return {"token": token, **result}


def safe_analyze_metric_cache(
    task: tuple[str, str, InteractionThresholds],
) -> dict[str, object]:
    token, raw_cache_path, thresholds = task
    try:
        return {
            "ok": True,
            "token": token,
            "result": analyze_metric_cache(token, Path(raw_cache_path), thresholds),
        }
    except Exception as exc:
        return {"ok": False, "token": token, "error": repr(exc)}
