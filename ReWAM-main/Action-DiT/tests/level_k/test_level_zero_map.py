"""Tests for causal map-lane Level-0 reference geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from navsim.agents.videodrive.level_k.level_zero_map import (
    extend_lane_centerline,
    project_and_sample_centerline,
    select_best_lane,
    select_minimum_heading_successor,
)


@dataclass
class _State:
    x: float
    y: float
    heading: float

    @property
    def array(self) -> np.ndarray:
        return np.asarray([self.x, self.y, self.heading], dtype=np.float64)


class _Baseline:
    def __init__(self, points) -> None:
        self.discrete_path = [_State(*point) for point in points]


class _Edge:
    def __init__(self, edge_id: str, points) -> None:
        self.id = edge_id
        self.baseline_path = _Baseline(points)
        self.outgoing_edges = []


def test_project_and_sample_centerline_follows_curve() -> None:
    path = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, np.pi / 2]],
        dtype=np.float64,
    )
    sampled = project_and_sample_centerline(
        path,
        start_xy=np.asarray([1.0, 0.2]),
        travel_distances=np.asarray([0.5, 1.5, 2.5]),
    )
    assert sampled is not None
    np.testing.assert_allclose(sampled[:, :2], [[1.5, 0.0], [2.0, 0.5], [2.0, 1.5]])
    np.testing.assert_allclose(sampled[:, 2], [0.0, np.pi / 2, np.pi / 2])


def test_project_and_sample_rejects_incomplete_horizon() -> None:
    sampled = project_and_sample_centerline(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        start_xy=np.asarray([0.5, 0.0]),
        travel_distances=np.asarray([1.0]),
    )
    assert sampled is None


def test_successor_selection_prefers_smallest_heading_change() -> None:
    root = _Edge("root", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    straight = _Edge("straight", [(1.0, 0.0, 0.0), (2.0, 0.1, 0.0)])
    turn = _Edge("turn", [(1.0, 0.0, 0.0), (1.0, 1.0, np.pi / 2)])
    root.outgoing_edges = [turn, straight]
    assert select_minimum_heading_successor(root) is straight


def test_extend_lane_centerline_uses_successors() -> None:
    first = _Edge("a", [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    second = _Edge("b", [(2.0, 0.0, 0.0), (5.0, 0.0, 0.0)])
    first.outgoing_edges = [second]
    path = extend_lane_centerline(first, required_length=4.0, max_depth=3)
    np.testing.assert_allclose(path[:, :2], [[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]])


def test_lane_matching_checks_distance_and_heading() -> None:
    aligned = _Edge("aligned", [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)])
    opposite = _Edge("opposite", [(10.0, 1.0, np.pi), (0.0, 1.0, np.pi)])
    selected = select_best_lane(
        [opposite, aligned],
        pose=np.asarray([2.0, 0.2, 0.0]),
        max_lateral_error=2.0,
        max_heading_error=np.pi / 3,
    )
    assert selected is aligned
