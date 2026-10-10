from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from scripts.visualization.analyze_navtest_performance_by_interaction import (
    assign_interaction_bins,
)
from scripts.visualization.navtest_interaction_intensity import (
    InteractionThresholds,
    aggregate_scene_interaction,
    analyze_agent_geometry,
)


def _polygon(center_x: float, center_y: float):
    return box(
        center_x - 1.0,
        center_y - 0.5,
        center_x + 1.0,
        center_y + 0.5,
    )


def _crossing_agent(crossing_time_index: int):
    times = np.arange(0.0, 2.5, 0.5)
    ego = [_polygon(2.0 * index, 0.0) for index in range(len(times))]
    agent = [
        _polygon(4.0, 2.0 * (index - crossing_time_index))
        for index in range(len(times))
    ]
    return analyze_agent_geometry(
        track_token=f"agent-{crossing_time_index}",
        tracked_object_type="vehicle",
        times_s=times,
        ego_polygons=ego,
        agent_polygons=agent,
        thresholds=InteractionThresholds(),
    )


def test_simultaneous_crossing_is_stronger_than_delayed_crossing() -> None:
    simultaneous = _crossing_agent(crossing_time_index=2)
    delayed = _crossing_agent(crossing_time_index=4)

    assert simultaneous.eligible
    assert delayed.eligible
    assert simultaneous.conflict_time_gap_s < delayed.conflict_time_gap_s
    assert simultaneous.interaction_intensity_v2 > delayed.interaction_intensity_v2


def test_far_agent_is_not_an_interaction() -> None:
    times = np.arange(0.0, 2.5, 0.5)
    ego = [_polygon(float(index), 0.0) for index in range(len(times))]
    agent = [_polygon(60.0 + index, 20.0) for index in range(len(times))]
    result = analyze_agent_geometry(
        track_token="far-agent",
        tracked_object_type="vehicle",
        times_s=times,
        ego_polygons=ego,
        agent_polygons=agent,
        thresholds=InteractionThresholds(),
    )
    scene = aggregate_scene_interaction(
        [result],
        reference_initial_speed_mps=5.0,
        reference_min_speed_mps=5.0,
        thresholds=InteractionThresholds(),
    )

    assert not result.eligible
    assert scene["interacting_vehicle_count"] == 0
    assert scene["interaction_intensity_v2"] == 0.0


def test_multiple_conflict_agents_add_bounded_bonus() -> None:
    first = _crossing_agent(crossing_time_index=2)
    second = replace(first, track_token="second-agent")
    thresholds = InteractionThresholds()
    one = aggregate_scene_interaction(
        [first],
        reference_initial_speed_mps=5.0,
        reference_min_speed_mps=3.0,
        thresholds=thresholds,
    )
    two = aggregate_scene_interaction(
        [first, second],
        reference_initial_speed_mps=5.0,
        reference_min_speed_mps=3.0,
        thresholds=thresholds,
    )

    assert two["interaction_intensity_v2"] == pytest.approx(
        min(float(one["interaction_intensity_v2"]) + 0.05, 1.0)
    )
    assert two["required_response_score"] == one["required_response_score"]


def test_reference_response_is_separate_from_interaction_intensity() -> None:
    agent = _crossing_agent(crossing_time_index=2)
    thresholds = InteractionThresholds()
    steady = aggregate_scene_interaction(
        [agent],
        reference_initial_speed_mps=5.0,
        reference_min_speed_mps=5.0,
        thresholds=thresholds,
    )
    braking = aggregate_scene_interaction(
        [agent],
        reference_initial_speed_mps=5.0,
        reference_min_speed_mps=1.0,
        thresholds=thresholds,
    )

    assert braking["required_response_score"] > steady["required_response_score"]
    assert braking["interaction_intensity_v2"] == steady["interaction_intensity_v2"]


def test_assign_interaction_bins_respects_frozen_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "interacting_vehicle_count": [0, 1, 1, 1, 2],
            "interaction_intensity_v2": [0.0, 0.1, 0.3, 0.6, 0.9],
        }
    )
    bins = assign_interaction_bins(
        frame,
        {"weak_max": 0.2, "medium_max": 0.5, "strong_max": 0.8},
    )

    assert bins.tolist() == [
        "none",
        "weak",
        "medium",
        "strong",
        "critical",
    ]


def test_threshold_validation_rejects_non_positive_scales() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        InteractionThresholds(proximity_scale_m=0.0).validate()
