"""Run one VideoDrive sample and render trajectory diagnostics.

This utility intentionally reruns inference because the standard PDM result
CSV stores scores, not the predicted trajectory.  Oracle Level-k inference is
kept on the production path by passing the current Scene to the same feature
builder used by ``VideoDriveAgent.compute_trajectory``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from matplotlib.lines import Line2D
from nuplan.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)
from PIL import Image, ImageDraw

from navsim.agents.videodrive.level_k import EgoTrajectoryCodec
from navsim.agents.videodrive.videodrive_agent import VideoDriveAgent
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, Trajectory
from navsim.visualization.bev import add_trajectory_to_bev_ax
from navsim.visualization.camera import (
    add_camera_ax,
    add_trajectory_to_camera_ax,
)
from navsim.visualization.config import TRAJECTORY_CONFIG
from navsim.visualization.plots import plot_bev_frame


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Defaults to DRIVELAW_MODEL_CONFIG.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Defaults to OPENSCENE_DATA_ROOT.",
    )
    parser.add_argument(
        "--data-split",
        default="test",
        help="NAVSIM storage split. navtest uses the 'test' directory.",
    )
    parser.add_argument(
        "--evaluation-csv",
        type=Path,
        default=None,
        help="Optional PDM CSV used to attach the token's metrics.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def required_path(
    explicit: Path | None,
    *,
    environment_name: str,
    description: str,
) -> Path:
    """Resolve and validate an explicit or environment-provided path."""

    value = explicit
    if value is None:
        environment_value = os.environ.get(environment_name)
        if environment_value:
            value = Path(environment_value)
    if value is None:
        raise ValueError(
            f"{description} is required. Pass it explicitly or set "
            f"{environment_name}."
        )
    value = value.expanduser().resolve()
    if not value.exists():
        raise FileNotFoundError(f"{description} does not exist: {value}")
    return value


def load_metric_row(csv_path: Path | None, token: str) -> dict[str, Any]:
    """Read one result row without requiring pandas."""

    if csv_path is None:
        return {}
    if not csv_path.is_file():
        raise FileNotFoundError(f"Evaluation CSV does not exist: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if str(row.get("token")) == token:
                return dict(row)
    raise KeyError(f"Token {token} is not present in {csv_path}.")


def run_inference_with_level_k_output(
    agent: VideoDriveAgent,
    agent_input: Any,
    scene: Any,
) -> tuple[Trajectory, Any, dict[str, torch.Tensor]]:
    """Use the production feature path while retaining Level-k trajectories."""

    features: dict[str, torch.Tensor] = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(agent_input))

    if agent.level_k_config.enabled:
        if not agent.level_k_config.oracle_agent_states:
            raise NotImplementedError(
                "This diagnostic currently expects Oracle current-agent states."
            )
        features.update(
            agent._surround_agent_builder.compute_current_inputs(scene)
        )

    batched_features = {
        name: value.unsqueeze(0)
        for name, value in features.items()
    }
    agent.eval()
    with torch.no_grad():
        predictions = agent.forward(batched_features)

    poses = np.asarray(predictions["actions"], dtype=np.float32)
    trajectory = Trajectory(
        poses=poses,
        trajectory_sampling=agent._trajectory_sampling,
    )
    return trajectory, predictions.get("level_k_output"), features


def run_video_dit_inference(
    agent: VideoDriveAgent,
    features: dict[str, torch.Tensor],
) -> list[Image.Image]:
    """Generate decoded Video DiT frames in a separate video-only pass.

    The production Level-k action path intentionally calls the diffusion
    pipeline with ``return_video=False``.  A diagnostic video therefore needs
    a second pass through the same Video DiT/VAE while omitting the Level-k
    action-only arguments.  The conditioning frames, text prompt, state
    context, diffusion seed and inference settings remain identical to
    ``VideoDriveAgent.forward_test``.
    """

    device = agent.device
    args = agent.args
    weight_dtype = torch.bfloat16
    cond_tensor = features["images"]
    if cond_tensor.ndim == 5:
        cond_tensor = cond_tensor.squeeze(0)
    elif cond_tensor.ndim == 3:
        cond_tensor = cond_tensor.unsqueeze(0)
    elif cond_tensor.ndim != 4:
        raise ValueError(
            "Expected Video DiT conditioning frames with shape (T,C,H,W), "
            f"got {tuple(cond_tensor.shape)}."
        )

    def _as_batch(value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(0)

    context_dict = {
        "cmd_onehot": _as_batch(features["driving_command"]).to(
            device,
            dtype=weight_dtype,
        ),
        "vel": _as_batch(features["vel"]).to(device, dtype=weight_dtype),
        "acc": _as_batch(features["acc"]).to(device, dtype=weight_dtype),
    }
    prompt = (
        "A high-quality, photorealistic dashboard camera view of autonomous "
        "driving. Based on the past 2 seconds videos, predict and generate "
        "the next 4 seconds of realistic driving continuation, Maintain "
        "temporal consistency, stable camera perspective, natural motion "
        "flow without jitter or artifacts, clear details, and realistic "
        "physics."
    )
    negative_prompt = getattr(
        args,
        "negative_prompt",
        "worst quality, low quality, blurry, jittery, distorted, motion "
        "blur, ghosting, flickering, stuttering, camera shake, unstable "
        "footage, warping, trailing artifacts, temporal inconsistency, "
        "jerky motion, choppy framerate",
    )
    if agent.view_mode == "surround6":
        height, width = 256, 3072
    else:
        height = int(getattr(args, "height", 768) or 768)
        width = int(getattr(args, "width", 1344) or 1344)

    condition = LTXVideoCondition(
        video=agent._tensor_video_to_pils(cond_tensor),
        frame_index=0,
    )
    seed = int(getattr(args, "seed", 42))
    generator = torch.Generator(device=device).manual_seed(seed)
    output = agent.pipe.infer(
        conditions=[condition],
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=10,
        guidance_scale=1.0,
        height=height,
        width=width,
        generator=generator,
        num_frames=18,
        output_type="pil",
        return_action=False,
        return_video=True,
        context_dict=context_dict,
        noise_seed=42,
        pixel_wise_timestep=True,
        image_cond_noise_scale=0.0,
    )
    frames: Any = output.frames if hasattr(output, "frames") else output
    if isinstance(frames, (list, tuple)) and len(frames) == 1 and isinstance(
        frames[0],
        (list, tuple),
    ):
        frames = frames[0]
    if not isinstance(frames, (list, tuple)) or not frames:
        raise TypeError(
            "Video DiT did not return a non-empty PIL frame sequence; "
            f"received {type(frames)!r}."
        )

    normalized_frames: list[Image.Image] = []
    for index, frame in enumerate(frames):
        if isinstance(frame, Image.Image):
            normalized_frames.append(frame.convert("RGB"))
        elif isinstance(frame, np.ndarray):
            array = frame
            if np.issubdtype(array.dtype, np.floating):
                array = np.clip(array, 0.0, 1.0) * 255.0
            normalized_frames.append(
                Image.fromarray(array.astype(np.uint8)).convert("RGB")
            )
        else:
            raise TypeError(
                f"Unsupported Video DiT frame {index} type: {type(frame)!r}."
            )
    return normalized_frames


def save_video_dit_images(
    output_dir: Path,
    frames: list[Image.Image],
) -> dict[str, Path]:
    """Save every decoded frame, the last frame and a compact contact sheet."""

    if not frames:
        raise ValueError("At least one Video DiT frame is required.")
    frames_dir = output_dir / "video_dit_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.save(frames_dir / f"frame_{index:03d}.png")

    last_frame_path = output_dir / "video_dit_last_frame.png"
    frames[-1].save(last_frame_path)

    thumbnail_width = min(448, frames[0].width)
    thumbnail_height = max(
        1,
        round(frames[0].height * thumbnail_width / frames[0].width),
    )
    columns = min(3, len(frames))
    rows = math.ceil(len(frames) / columns)
    label_height = 24
    montage = Image.new(
        "RGB",
        (columns * thumbnail_width, rows * (thumbnail_height + label_height)),
        color="white",
    )
    draw = ImageDraw.Draw(montage)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        thumbnail = frame.resize(
            (thumbnail_width, thumbnail_height),
            resample=Image.Resampling.LANCZOS,
        )
        x = column * thumbnail_width
        y = row * (thumbnail_height + label_height)
        montage.paste(thumbnail, (x, y))
        draw.text((x + 6, y + thumbnail_height + 4), f"frame {index:02d}", fill="black")

    montage_path = output_dir / "video_dit_output.png"
    montage.save(montage_path)
    return {
        "video_dit_montage": montage_path,
        "video_dit_last_frame": last_frame_path,
        "video_dit_frames_dir": frames_dir,
    }


def finite_difference_kinematics(
    poses: np.ndarray,
    *,
    interval_length: float,
) -> dict[str, np.ndarray]:
    """Compute transparent finite-difference diagnostics from local poses."""

    origin_xy = np.zeros((1, 2), dtype=np.float64)
    xy = np.concatenate((origin_xy, poses[:, :2]), axis=0)
    displacement = np.diff(xy, axis=0)
    speed = np.linalg.norm(displacement, axis=1) / interval_length

    speed_with_initial = np.concatenate(([speed[0]], speed))
    acceleration = np.diff(speed_with_initial) / interval_length
    acceleration_with_initial = np.concatenate(
        ([acceleration[0]], acceleration)
    )
    jerk = np.diff(acceleration_with_initial) / interval_length

    heading = np.unwrap(
        np.concatenate(([0.0], poses[:, 2].astype(np.float64)))
    )
    yaw_rate = np.diff(heading) / interval_length
    return {
        "speed": speed,
        "acceleration": acceleration,
        "jerk": jerk,
        "yaw_rate": yaw_rate,
    }


def save_trajectory_table(
    output_path: Path,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    predicted_kinematics: dict[str, np.ndarray],
    ground_truth_kinematics: dict[str, np.ndarray],
    *,
    interval_length: float,
) -> None:
    """Save auditable per-timestep poses and diagnostics."""

    fieldnames = [
        "time_s",
        "pred_x_m",
        "pred_y_m",
        "pred_heading_rad",
        "gt_x_m",
        "gt_y_m",
        "gt_heading_rad",
        "pred_speed_mps",
        "pred_acceleration_mps2",
        "pred_jerk_mps3",
        "pred_yaw_rate_radps",
        "gt_speed_mps",
        "gt_acceleration_mps2",
        "gt_jerk_mps3",
        "gt_yaw_rate_radps",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(predicted.shape[0]):
            row = {
                "time_s": (index + 1) * interval_length,
                "pred_x_m": predicted[index, 0],
                "pred_y_m": predicted[index, 1],
                "pred_heading_rad": predicted[index, 2],
                "gt_x_m": ground_truth[index, 0],
                "gt_y_m": ground_truth[index, 1],
                "gt_heading_rad": ground_truth[index, 2],
            }
            for prefix, values in (
                ("pred", predicted_kinematics),
                ("gt", ground_truth_kinematics),
            ):
                row[f"{prefix}_speed_mps"] = values["speed"][index]
                row[f"{prefix}_acceleration_mps2"] = values[
                    "acceleration"
                ][index]
                row[f"{prefix}_jerk_mps3"] = values["jerk"][index]
                row[f"{prefix}_yaw_rate_radps"] = values["yaw_rate"][index]
            writer.writerow(row)


def plot_bev_diagnostic(
    output_path: Path,
    scene: Any,
    predicted: Trajectory,
    ground_truth: Trajectory,
    *,
    token: str,
    metric_row: dict[str, Any],
    level_k_output: Any,
    features: dict[str, torch.Tensor],
    selected_track_tokens: tuple[str, ...],
) -> None:
    """Render current BEV, Ego prediction/GT, and Level-k agent paths."""

    level_k_color = "#FFD400"
    ground_truth_color = "#1F77B4"
    frame_index = scene.scene_metadata.num_history_frames - 1
    figure, axis = plot_bev_frame(scene, frame_index)

    gt_config = dict(TRAJECTORY_CONFIG["human"])
    gt_config.update(
        {
            "line_color": ground_truth_color,
            "fill_color": ground_truth_color,
        }
    )
    prediction_config = dict(TRAJECTORY_CONFIG["agent"])
    prediction_config.update(
        {"line_color": level_k_color, "fill_color": level_k_color}
    )
    add_trajectory_to_bev_ax(axis, ground_truth, gt_config)
    add_trajectory_to_bev_ax(axis, predicted, prediction_config)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=level_k_color,
            marker="o",
            label=f"Ego Level-{level_k_output.reasoning_level}"
            if level_k_output is not None
            else "Ego prediction",
        ),
        Line2D(
            [0],
            [0],
            color=ground_truth_color,
            marker="o",
            label="Ego GT",
        ),
    ]

    if level_k_output is not None:
        ego_codec = EgoTrajectoryCodec()
        level_zero = (
            ego_codec.strategy_to_physical(level_k_output.ego_levels[0])
            .detach()
            .float()
            .cpu()
            .numpy()[0]
        )
        level_zero_xy = np.concatenate(
            (np.zeros((1, 2)), level_zero[:, :2]),
            axis=0,
        )
        axis.plot(
            level_zero_xy[:, 1],
            level_zero_xy[:, 0],
            color=level_k_color,
            linestyle="--",
            linewidth=1.5,
            label="Ego Level-0",
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=level_k_color,
                linestyle="--",
                label="Ego Level-0",
            )
        )

        level_zero_other = (
            level_k_output.other_levels[0]
            .detach()
            .float()
            .cpu()
            .numpy()[0]
        )
        final_other = (
            level_k_output.other_levels[level_k_output.reasoning_level]
            .detach()
            .float()
            .cpu()
            .numpy()[0]
        )
        current_states = (
            features["agent_current_states"].detach().cpu().numpy()
        )
        agent_mask = features["agent_mask"].detach().cpu().numpy().astype(bool)
        for slot in np.flatnonzero(agent_mask):
            origin = current_states[slot, :2]
            level_zero_other_xy = np.concatenate(
                (
                    origin[None, :],
                    origin[None, :] + level_zero_other[slot, :, :2],
                ),
                axis=0,
            )
            other_xy = np.concatenate(
                (
                    origin[None, :],
                    origin[None, :] + final_other[slot, :, :2],
                ),
                axis=0,
            )
            axis.plot(
                level_zero_other_xy[:, 1],
                level_zero_other_xy[:, 0],
                color=level_k_color,
                linestyle="--",
                linewidth=1.1,
                alpha=0.65,
            )
            axis.plot(
                other_xy[:, 1],
                other_xy[:, 0],
                color=level_k_color,
                linewidth=1.3,
                alpha=0.9,
            )
            label = (
                selected_track_tokens[slot][:8]
                if slot < len(selected_track_tokens)
                else f"slot-{slot}"
            )
            axis.text(
                other_xy[-1, 1],
                other_xy[-1, 0],
                label,
                color="#9A7D00",
                fontsize=6,
            )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=level_k_color,
                linestyle="--",
                linewidth=1.1,
                label="Other Level-0",
            )
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=level_k_color,
                linestyle="-",
                linewidth=1.3,
                label=f"Other Level-{level_k_output.reasoning_level}",
            )
        )

    score = metric_row.get("score")
    comfort = metric_row.get("comfort")
    subtitle = ""
    if score not in (None, ""):
        subtitle = f"\nscore={float(score):.4f}, comfort={float(comfort):.4f}"
    axis.set_title(f"Token {token}{subtitle}")
    axis.legend(handles=legend_handles, loc="best", fontsize=7)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_front_camera_diagnostic(
    output_path: Path,
    scene: Any,
    predicted: Trajectory,
    ground_truth: Trajectory,
    *,
    token: str,
) -> None:
    """Project predicted and GT trajectories onto the current front camera."""

    frame_index = scene.scene_metadata.num_history_frames - 1
    camera = scene.frames[frame_index].cameras.cam_f0
    required_camera_fields = {
        "image": camera.image,
        "sensor2lidar_rotation": camera.sensor2lidar_rotation,
        "sensor2lidar_translation": camera.sensor2lidar_translation,
        "intrinsics": camera.intrinsics,
    }
    missing_fields = [
        name for name, value in required_camera_fields.items() if value is None
    ]
    if missing_fields:
        raise ValueError(
            "Front-camera trajectory projection requires: "
            + ", ".join(missing_fields)
        )

    image_height, image_width = camera.image.shape[:2]
    figure_width = 12.0
    figure_height = max(4.0, figure_width * image_height / image_width)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    add_camera_ax(axis, camera)

    predicted_config = {
        **TRAJECTORY_CONFIG["agent"],
        "line_color": "#D62728",
        "line_width": 4.0,
        "marker": None,
        "arrow_color": "#D62728",
        "arrow_edge_color": "#D62728",
        "arrow_alpha": 1.0,
        "arrow_line_width": 2.0,
    }
    ground_truth_config = {
        **TRAJECTORY_CONFIG["human"],
        "line_color": "#2CA02C",
        "line_width": 4.0,
        "marker": None,
        "arrow_color": "#2CA02C",
        "arrow_edge_color": "#2CA02C",
        "arrow_alpha": 1.0,
        "arrow_line_width": 2.0,
    }
    add_trajectory_to_camera_ax(axis, camera, predicted, predicted_config)
    add_trajectory_to_camera_ax(axis, camera, ground_truth, ground_truth_config)

    axis.set_title(f"Front camera trajectory projection - token {token}")
    axis.legend(
        handles=(
            Line2D([0], [0], color="#D62728", linewidth=4, label="Prediction"),
            Line2D([0], [0], color="#2CA02C", linewidth=4, label="GT"),
        ),
        loc="upper right",
    )
    axis.set_xlim(0, image_width)
    axis.set_ylim(image_height, 0)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)


def plot_kinematics(
    output_path: Path,
    predicted: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
    *,
    interval_length: float,
    token: str,
) -> None:
    """Render speed, acceleration, jerk and yaw-rate comparisons."""

    num_steps = predicted["speed"].shape[0]
    times = np.arange(1, num_steps + 1) * interval_length
    panels = (
        ("speed", "Longitudinal speed (m/s)"),
        ("acceleration", "Acceleration [m/s²]"),
        ("jerk", "Jerk [m/s³]"),
        ("yaw_rate", "Yaw rate [rad/s]"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, (name, label) in zip(axes.flat, panels):
        axis.plot(times, predicted[name], "o-", color="#D62728", linewidth=2.5, label="Prediction")
        axis.plot(times, ground_truth[name], "o-", color="#2CA02C", linewidth=2.5, label="GT")
        axis.axhline(0.0, color="#888888", linewidth=1.2)
        axis.set_ylabel(label, fontsize=16)
        axis.tick_params(axis="both", labelsize=16)
        axis.grid(True, alpha=0.25)
        axis.legend()
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)", fontsize=16)
    figure.suptitle(
        f"Finite-difference kinematics — token {token}\n"
        "(diagnostic only; not the internal PDM comfort calculation)"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run one deterministic sample and write all requested diagnostics."""

    args = parse_args()
    model_config = required_path(
        args.model_config,
        environment_name="DRIVELAW_MODEL_CONFIG",
        description="DriveLaW model config",
    )
    data_root = required_path(
        args.data_root,
        environment_name="OPENSCENE_DATA_ROOT",
        description="OpenScene data root",
    )
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for VideoDrive inference.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_row = load_metric_row(args.evaluation_csv, args.token)
    sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    agent = VideoDriveAgent(
        trajectory_sampling=sampling,
        config_file=str(model_config),
        weight_dtype="",
        device=args.device,
    )
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        tokens=[args.token],
    )
    scene_loader = SceneLoader(
        data_path=data_root / "navsim_logs" / args.data_split,
        sensor_blobs_path=data_root / "sensor_blobs" / args.data_split,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=False,
    )
    if args.token not in scene_loader.tokens:
        raise KeyError(
            f"Token {args.token} was not found in data split "
            f"{args.data_split!r}."
        )

    scene = scene_loader.get_scene_from_token(args.token)
    agent_input = scene_loader.get_agent_input_from_token(args.token)
    agent.initialize()
    prediction, level_k_output, features = run_inference_with_level_k_output(
        agent,
        agent_input,
        scene,
    )
    ground_truth = scene.get_future_trajectory(
        num_trajectory_frames=prediction.poses.shape[0]
    )

    interval = prediction.trajectory_sampling.interval_length
    predicted_kinematics = finite_difference_kinematics(
        prediction.poses,
        interval_length=interval,
    )
    ground_truth_kinematics = finite_difference_kinematics(
        ground_truth.poses,
        interval_length=interval,
    )
    trajectory_csv = args.output_dir / "trajectory_comparison.csv"
    bev_png = args.output_dir / "trajectory_bev.png"
    front_camera_png = args.output_dir / "trajectory_front_camera.png"
    kinematics_png = args.output_dir / "kinematics.png"
    level_k_npz = args.output_dir / "level_k_trajectories.npz"
    summary_json = args.output_dir / "summary.json"

    save_trajectory_table(
        trajectory_csv,
        prediction.poses,
        ground_truth.poses,
        predicted_kinematics,
        ground_truth_kinematics,
        interval_length=interval,
    )
    selected_track_tokens: tuple[str, ...] = ()
    if agent.level_k_config.enabled:
        selected_track_tokens = (
            agent._surround_agent_builder.selected_track_tokens(scene)
        )
    plot_bev_diagnostic(
        bev_png,
        scene,
        prediction,
        ground_truth,
        token=args.token,
        metric_row=metric_row,
        level_k_output=level_k_output,
        features=features,
        selected_track_tokens=selected_track_tokens,
    )
    plot_front_camera_diagnostic(
        front_camera_png,
        scene,
        prediction,
        ground_truth,
        token=args.token,
    )
    plot_kinematics(
        kinematics_png,
        predicted_kinematics,
        ground_truth_kinematics,
        interval_length=interval,
        token=args.token,
    )
    video_dit_frames = run_video_dit_inference(agent, features)
    video_dit_outputs = save_video_dit_images(
        args.output_dir,
        video_dit_frames,
    )

    arrays: dict[str, np.ndarray] = {
        "ego_prediction": prediction.poses,
        "ego_ground_truth": ground_truth.poses,
        "agent_current_states": features.get(
            "agent_current_states",
            torch.empty(0),
        ).detach().cpu().numpy(),
        "agent_mask": features.get(
            "agent_mask",
            torch.empty(0, dtype=torch.bool),
        ).detach().cpu().numpy(),
        "track_tokens": np.asarray(selected_track_tokens, dtype=str),
    }
    if level_k_output is not None:
        for level, trajectory in enumerate(level_k_output.ego_levels):
            arrays[f"ego_level_{level}"] = (
                trajectory.detach().float().cpu().numpy()
            )
        for level, trajectory in enumerate(level_k_output.other_levels):
            arrays[f"other_level_{level}"] = (
                trajectory.detach().float().cpu().numpy()
            )
    np.savez_compressed(level_k_npz, **arrays)

    summary = {
        "token": args.token,
        "model_config": str(model_config),
        "diffusion_weights": os.environ.get("DRIVELAW_DIFFUSION_WEIGHTS"),
        "data_split": args.data_split,
        "reasoning_level": agent.level_k_config.reasoning_level,
        "oracle_agent_states": agent.level_k_config.oracle_agent_states,
        "video_dit_num_frames": len(video_dit_frames),
        "video_dit_generation": (
            "Separate video-only pass with the same Video DiT, conditioning "
            "frames, prompt, context and seed; Level-k action recursion is "
            "not invoked in this decoding pass."
        ),
        "evaluation_metrics": metric_row,
        "outputs": {
            "trajectory_csv": str(trajectory_csv),
            "bev_png": str(bev_png),
            "front_camera_png": str(front_camera_png),
            "kinematics_png": str(kinematics_png),
            "level_k_npz": str(level_k_npz),
            **{
                name: str(path)
                for name, path in video_dit_outputs.items()
            },
        },
    }
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Single-sample VideoDrive diagnostics completed.")
    for output in (
        trajectory_csv,
        bev_png,
        front_camera_png,
        kinematics_png,
        level_k_npz,
        *video_dit_outputs.values(),
        summary_json,
    ):
        print(output)


if __name__ == "__main__":
    main()
