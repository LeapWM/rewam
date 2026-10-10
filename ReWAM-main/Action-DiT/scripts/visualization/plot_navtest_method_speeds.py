#!/usr/bin/env python3
"""Plot finite-difference speeds for saved NAVTEST method trajectories and GT."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import fontManager


TOKEN_PATTERN = re.compile(r"^[0-9a-fA-F]{16}$")
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
INTERVAL_LENGTH_S = 0.5
EXPECTED_POSE_SHAPE = (8, 3)
LINE_WIDTH_PT = 2.5
FONT_SIZE_PT = 16
REQUESTED_FONT = "Arial"
RESOLVED_FONT = (
    REQUESTED_FONT
    if any(font.name == REQUESTED_FONT for font in fontManager.ttflist)
    else "DejaVu Sans"
)
KNOWN_STYLES = {
    "level_k_wam": ("#D62728", "o", "Level-K WAM"),
    "drivelaw": ("#1F77B4", "s", "DriveLaw"),
    "diffusiondrive": ("#FF7F0E", "*", "DiffusionDrive"),
}
FALLBACK_STYLES = (
    ("#9467BD", "h"),
    ("#8C564B", "p"),
    ("#17BECF", "v"),
)
GT_STYLE = ("#2CA02C", "D", "GT")


@dataclass(frozen=True)
class MethodSpec:
    name: str
    root: Path


def parse_method(value: str) -> MethodSpec:
    name, separator, raw_root = value.partition("=")
    name = name.strip()
    raw_root = raw_root.strip()
    if not separator or not LABEL_PATTERN.fullmatch(name) or not raw_root:
        raise argparse.ArgumentTypeError(
            "--method must use a valid NAME=RESULT_ROOT value"
        )
    return MethodSpec(name=name, root=Path(raw_root).expanduser())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", required=True)
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        type=parse_method,
        metavar="NAME=RESULT_ROOT",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--interval-length", type=float, default=INTERVAL_LENGTH_S)
    parser.add_argument("--line-width", type=float, default=LINE_WIDTH_PT)
    parser.add_argument("--font-size", type=float, default=FONT_SIZE_PT)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show-legend", action="store_true")
    return parser.parse_args()


def resolve_token_dir(root: Path, token: str) -> Path:
    root = root.expanduser().resolve()
    candidates = []
    direct = root / "tokens" / token
    if direct.is_dir():
        candidates.append(direct)
    candidates.extend(
        path
        for path in sorted(root.glob(f"shard_*/tokens/{token}"))
        if path.is_dir()
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one token directory below {root}; "
            f"token={token}, found={len(candidates)}"
        )
    return candidates[0]


def load_trajectories(
    methods: Sequence[MethodSpec], token: str
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    canonical_gt: np.ndarray | None = None
    for method in methods:
        trajectory_path = resolve_token_dir(method.root, token) / "trajectory.npz"
        if not trajectory_path.is_file():
            raise FileNotFoundError(f"Missing trajectory: {trajectory_path}")
        with np.load(trajectory_path, allow_pickle=False) as payload:
            if "prediction" not in payload or "ground_truth" not in payload:
                raise ValueError(
                    f"{trajectory_path} must contain prediction and ground_truth"
                )
            prediction = np.asarray(payload["prediction"], dtype=np.float64)
            ground_truth = np.asarray(payload["ground_truth"], dtype=np.float64)
        if prediction.shape != EXPECTED_POSE_SHAPE:
            raise ValueError(
                f"Unexpected prediction shape for {method.name}: {prediction.shape}"
            )
        if ground_truth.shape != EXPECTED_POSE_SHAPE:
            raise ValueError(
                f"Unexpected GT shape for {method.name}: {ground_truth.shape}"
            )
        if not np.isfinite(prediction).all() or not np.isfinite(ground_truth).all():
            raise ValueError(f"Non-finite poses for {method.name}")
        if canonical_gt is None:
            canonical_gt = ground_truth.copy()
        elif not np.allclose(canonical_gt, ground_truth, atol=1.0e-5, rtol=0.0):
            raise ValueError(f"Ground-truth mismatch for method {method.name}")
        predictions[method.name] = prediction
    assert canonical_gt is not None
    return predictions, canonical_gt


def finite_difference_speed(poses: np.ndarray, interval_length: float) -> np.ndarray:
    xy = np.concatenate(
        [np.zeros((1, 2), dtype=np.float64), poses[:, :2]], axis=0
    )
    return np.linalg.norm(np.diff(xy, axis=0), axis=1) / interval_length


def method_style(name: str, index: int) -> tuple[str, str, str]:
    known = KNOWN_STYLES.get(name.lower())
    if known is not None:
        return known
    color, marker = FALLBACK_STYLES[index % len(FALLBACK_STYLES)]
    return color, marker, name


def write_speed_csv(
    output_path: Path,
    times: np.ndarray,
    method_speeds: dict[str, np.ndarray],
    gt_speed: np.ndarray,
) -> None:
    fieldnames = ["time_s"] + [
        f"{name}_speed_mps" for name in method_speeds
    ] + ["gt_speed_mps"]
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, time_s in enumerate(times):
            row = {"time_s": float(time_s)}
            row.update(
                {
                    f"{name}_speed_mps": float(speed[index])
                    for name, speed in method_speeds.items()
                }
            )
            row["gt_speed_mps"] = float(gt_speed[index])
            writer.writerow(row)


def plot_speeds(
    output_path: Path,
    times: np.ndarray,
    methods: Sequence[MethodSpec],
    method_speeds: dict[str, np.ndarray],
    gt_speed: np.ndarray,
    *,
    line_width: float,
    font_size: float,
    dpi: int,
    show_legend: bool,
) -> None:
    with plt.rc_context(
        {
            "font.family": RESOLVED_FONT,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
        }
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        for index, method in enumerate(methods):
            color, marker, display_name = method_style(method.name, index)
            axis.plot(
                times,
                method_speeds[method.name],
                color=color,
                marker=marker,
                linewidth=line_width,
                markersize=6,
                label=display_name,
            )
        gt_color, gt_marker, gt_label = GT_STYLE
        axis.plot(
            times,
            gt_speed,
            color=gt_color,
            marker=gt_marker,
            linewidth=line_width,
            markersize=6,
            label=gt_label,
        )
        axis.set_xlabel("Time (s)", fontsize=font_size)
        axis.set_ylabel("Longitudinal speed (m/s)", fontsize=font_size)
        axis.tick_params(axis="both", labelsize=font_size)
        axis.grid(True, alpha=0.25)
        if show_legend:
            axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    args = parse_args()
    token = args.token.strip().lower()
    if not TOKEN_PATTERN.fullmatch(token):
        raise ValueError("--token must contain exactly 16 hexadecimal characters")
    methods = list(args.method)
    if len(methods) < 1:
        raise ValueError("At least one --method is required")
    if len({method.name for method in methods}) != len(methods):
        raise ValueError("Method names must be unique")
    if args.interval_length <= 0 or args.line_width <= 0 or args.font_size <= 0:
        raise ValueError("Interval, line width, and font size must be positive")
    if args.dpi <= 0:
        raise ValueError("DPI must be positive")

    predictions, ground_truth = load_trajectories(methods, token)
    method_speeds = {
        name: finite_difference_speed(poses, args.interval_length)
        for name, poses in predictions.items()
    }
    gt_speed = finite_difference_speed(ground_truth, args.interval_length)
    times = (
        np.arange(1, EXPECTED_POSE_SHAPE[0] + 1, dtype=np.float64)
        * args.interval_length
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "three_method_speed.csv"
    png_path = output_dir / "three_method_speed.png"
    write_speed_csv(csv_path, times, method_speeds, gt_speed)
    plot_speeds(
        png_path,
        times,
        methods,
        method_speeds,
        gt_speed,
        line_width=args.line_width,
        font_size=args.font_size,
        dpi=args.dpi,
        show_legend=args.show_legend,
    )
    print(f"Speed CSV: {csv_path}")
    print(f"Speed plot: {png_path}")
    print(f"Font: {RESOLVED_FONT}")


if __name__ == "__main__":
    main()
