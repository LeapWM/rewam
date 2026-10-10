"""SE(2) coordinate transforms used by surrounding-agent trajectory targets."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


PoseArray = npt.NDArray[np.float64]


def normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    angle_value = float(angle)
    if not np.isfinite(angle_value):
        raise ValueError("angle must be finite.")
    return float((angle_value + np.pi) % (2.0 * np.pi) - np.pi)


def _as_pose(pose: npt.ArrayLike, *, name: str) -> PoseArray:
    array = np.asarray(pose, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape [3] as (x, y, heading).")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _rotation(angle: float) -> PoseArray:
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.asarray(
        [[cos_angle, -sin_angle], [sin_angle, cos_angle]],
        dtype=np.float64,
    )


def local_pose_to_global(
    local_pose: npt.ArrayLike,
    ego_global_pose: npt.ArrayLike,
) -> PoseArray:
    """Transform an agent pose from a frame-local ego frame to world coordinates."""

    local = _as_pose(local_pose, name="local_pose")
    ego = _as_pose(ego_global_pose, name="ego_global_pose")

    global_xy = _rotation(float(ego[2])) @ local[:2] + ego[:2]
    global_heading = normalize_angle(float(local[2] + ego[2]))
    return np.asarray([global_xy[0], global_xy[1], global_heading], dtype=np.float64)


def global_pose_to_local(
    global_pose: npt.ArrayLike,
    ego_global_pose: npt.ArrayLike,
) -> PoseArray:
    """Transform a world-coordinate agent pose into a frame-local ego frame."""

    global_agent = _as_pose(global_pose, name="global_pose")
    ego = _as_pose(ego_global_pose, name="ego_global_pose")

    local_xy = _rotation(float(-ego[2])) @ (global_agent[:2] - ego[:2])
    local_heading = normalize_angle(float(global_agent[2] - ego[2]))
    return np.asarray([local_xy[0], local_xy[1], local_heading], dtype=np.float64)


def transform_pose_between_ego_frames(
    local_pose: npt.ArrayLike,
    source_ego_global_pose: npt.ArrayLike,
    target_ego_global_pose: npt.ArrayLike,
) -> PoseArray:
    """Express a source-frame local pose in a target-frame ego coordinate system."""

    global_pose = local_pose_to_global(local_pose, source_ego_global_pose)
    return global_pose_to_local(global_pose, target_ego_global_pose)


def transform_vector_between_ego_frames(
    local_vector: npt.ArrayLike,
    source_ego_heading: float,
    target_ego_heading: float,
) -> npt.NDArray[np.float64]:
    """Rotate a 2D vector between ego frames without applying translation."""

    vector = np.asarray(local_vector, dtype=np.float64)
    if vector.shape != (2,):
        raise ValueError("local_vector must have shape [2].")
    if not np.isfinite(vector).all():
        raise ValueError("local_vector contains non-finite values.")
    source_heading = float(source_ego_heading)
    target_heading = float(target_ego_heading)
    if not np.isfinite(source_heading):
        raise ValueError("source_ego_heading must be finite.")
    if not np.isfinite(target_heading):
        raise ValueError("target_ego_heading must be finite.")
    return _rotation(source_heading - target_heading) @ vector
