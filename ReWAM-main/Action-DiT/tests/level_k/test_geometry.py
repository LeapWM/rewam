"""Strict tests for the SE(2) transforms used by Level-k targets."""

from __future__ import annotations

import math

import numpy as np
import pytest

from navsim.agents.videodrive.level_k.geometry import (
    global_pose_to_local,
    local_pose_to_global,
    normalize_angle,
    transform_pose_between_ego_frames,
    transform_vector_between_ego_frames,
)


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, 0.0),
        (math.pi, -math.pi),
        (-math.pi, -math.pi),
        (3.0 * math.pi, -math.pi),
        (-3.0 * math.pi, -math.pi),
        (2.0 * math.pi + 0.25, 0.25),
    ],
)
def test_normalize_angle_wraps_to_half_open_interval(
    angle: float,
    expected: float,
) -> None:
    assert normalize_angle(angle) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("local_pose", "ego_pose"),
    [
        ([12.3, -4.5, math.pi - 1e-10], [100.0, -30.0, 1.2]),
        ([-8.0, 2.0, -math.pi + 1e-10], [-50.0, 80.0, -2.4]),
        ([0.0, 0.0, 0.0], [1e4, -1e4, math.pi / 2]),
    ],
)
def test_local_global_round_trip_with_translation_and_rotation(
    local_pose: list[float],
    ego_pose: list[float],
) -> None:
    recovered = global_pose_to_local(
        local_pose_to_global(local_pose, ego_pose),
        ego_pose,
    )
    np.testing.assert_allclose(recovered[:2], local_pose[:2], atol=1e-10)
    assert normalize_angle(recovered[2] - local_pose[2]) == pytest.approx(
        0.0,
        abs=1e-10,
    )


def test_pose_transform_matches_independent_known_answer() -> None:
    ego_global = np.asarray([10.0, 20.0, math.pi / 2])
    local_pose = np.asarray([2.0, 3.0, math.pi / 4])
    expected_global = np.asarray([7.0, 22.0, 3.0 * math.pi / 4])

    actual_global = local_pose_to_global(local_pose, ego_global)
    np.testing.assert_allclose(actual_global, expected_global, atol=1e-12)
    np.testing.assert_allclose(
        global_pose_to_local(expected_global, ego_global),
        local_pose,
        atol=1e-12,
    )


def test_between_frame_transform_matches_independent_known_answer() -> None:
    source_ego = np.asarray([10.0, 20.0, math.pi / 2])
    target_ego = np.asarray([13.0, 25.0, math.pi])
    source_local = np.asarray([2.0, 3.0, math.pi / 4])

    # The source-local pose is world pose (7, 22, 3π/4). Relative to the
    # target Ego at (13, 25, π), this is (6, 3, -π/4).
    expected_target_local = np.asarray([6.0, 3.0, -math.pi / 4])
    actual_target_local = transform_pose_between_ego_frames(
        source_local,
        source_ego,
        target_ego,
    )
    np.testing.assert_allclose(
        actual_target_local,
        expected_target_local,
        atol=1e-12,
    )


def test_between_frame_transform_is_invertible() -> None:
    source_ego = np.asarray([120.0, -35.0, 2.7])
    target_ego = np.asarray([-10.0, 88.0, -2.8])
    source_local = np.asarray([23.0, -7.0, math.pi - 1e-9])

    target_local = transform_pose_between_ego_frames(
        source_local,
        source_ego,
        target_ego,
    )
    recovered = transform_pose_between_ego_frames(
        target_local,
        target_ego,
        source_ego,
    )

    np.testing.assert_allclose(recovered[:2], source_local[:2], atol=1e-10)
    assert normalize_angle(recovered[2] - source_local[2]) == pytest.approx(
        0.0,
        abs=1e-10,
    )


def test_vector_transform_rotates_without_translation() -> None:
    vector = np.asarray([3.0, -2.0])
    transformed = transform_vector_between_ego_frames(
        vector,
        source_ego_heading=math.pi / 2,
        target_ego_heading=0.0,
    )
    np.testing.assert_allclose(transformed, [2.0, 3.0], atol=1e-12)

    same_headings = transform_vector_between_ego_frames(
        vector,
        source_ego_heading=-1.7,
        target_ego_heading=-1.7,
    )
    np.testing.assert_allclose(same_headings, vector, atol=1e-12)


def test_invalid_pose_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        local_pose_to_global([1.0, 2.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        global_pose_to_local(
            [float("nan"), 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )


@pytest.mark.parametrize("angle", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_angle_is_rejected(angle: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        normalize_angle(angle)


@pytest.mark.parametrize(
    ("source_heading", "target_heading", "expected_name"),
    [
        (float("nan"), 0.0, "source_ego_heading"),
        (0.0, float("inf"), "target_ego_heading"),
    ],
)
def test_non_finite_vector_frame_heading_is_rejected(
    source_heading: float,
    target_heading: float,
    expected_name: str,
) -> None:
    with pytest.raises(ValueError, match=expected_name):
        transform_vector_between_ego_frames(
            [1.0, 0.0],
            source_heading,
            target_heading,
        )
