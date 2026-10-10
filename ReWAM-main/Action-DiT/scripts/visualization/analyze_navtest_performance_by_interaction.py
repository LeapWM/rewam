#!/usr/bin/env python3
"""Analyze NAVTEST method performance across interaction-intensity bins."""

from __future__ import annotations

import argparse
import json
import lzma
import math
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)

try:
    from .navtest_interaction_intensity import (
        INTERACTION_DEFINITION_VERSION,
        InteractionThresholds,
        safe_analyze_metric_cache,
    )
except ImportError:
    from navtest_interaction_intensity import (
        INTERACTION_DEFINITION_VERSION,
        InteractionThresholds,
        safe_analyze_metric_cache,
    )


DEFAULT_METRIC_CACHE = Path("/mnt/c2-worldmodel/2639638/navsim_exp/metric_cache")
DEFAULT_SCENE_TYPES = Path(
    "/mnt/c2-worldmodel/2639638/navsim_exp/navtest_scene_types/"
    "gt4s_intersection_heading15deg_20260829/navtest_scene_types.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/dahuafs/userdata/2639638/navsim_exp/navtest_interaction_intensity"
)
METRIC_COLUMNS = {
    "NC": "no_at_fault_collisions",
    "DAC": "drivable_area_compliance",
    "TTC": "time_to_collision_within_bound",
    "EP": "ego_progress",
    "PDMS": "score",
}
BIN_ORDER = ("none", "weak", "medium", "strong", "critical")


def _parse_method(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    name = name.strip()
    raw_path = raw_path.strip()
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("--method must use NAME=RESULT_PATH")
    if not name.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError(f"Invalid method name: {name}")
    return name, Path(raw_path).expanduser()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        action="append",
        type=_parse_method,
        required=True,
        help="Method and NAVTEST result root as NAME=PATH; repeat as needed.",
    )
    parser.add_argument("--proposed-method", required=True)
    parser.add_argument("--metric-cache-path", type=Path, default=DEFAULT_METRIC_CACHE)
    parser.add_argument("--scene-types-csv", type=Path, default=DEFAULT_SCENE_TYPES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--interaction-metrics-csv",
        type=Path,
        help="Reuse previously extracted interaction geometry.",
    )
    parser.add_argument(
        "--fit-bin-interaction-csv",
        type=Path,
        help="Fit bin cutoffs on a separate interaction table, such as NAVTRAIN.",
    )
    parser.add_argument(
        "--bin-boundaries-json",
        type=Path,
        help="Apply frozen weak/medium/strong cutoffs from a previous run.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--rescore-missing-methods",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use trajectory.npz and the official PDM scorer when CSV metrics are absent.",
    )
    parser.add_argument(
        "--rescore-workers",
        type=int,
        help="Workers for official PDM rescoring; defaults to --workers.",
    )
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--horizon-s", type=float, default=4.0)
    parser.add_argument("--sample-interval-s", type=float, default=0.5)
    parser.add_argument("--nearby-radius-m", type=float, default=35.0)
    parser.add_argument("--min-agent-displacement-m", type=float, default=1.0)
    parser.add_argument("--min-agent-peak-speed-mps", type=float, default=0.5)
    parser.add_argument("--max-path-conflict-distance-m", type=float, default=3.0)
    parser.add_argument("--path-conflict-tolerance-m", type=float, default=0.25)
    parser.add_argument("--max-conflict-time-gap-s", type=float, default=1.5)
    parser.add_argument("--max-synchronized-distance-m", type=float, default=3.0)
    return parser.parse_args()


def _as_valid(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: (
            value is True
            or value == 1
            or str(value).strip().lower() in {"true", "1", "yes", "ok"}
        )
    )


def _metric_csvs(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        return [path]
    direct = path / "pdm_results.csv"
    if direct.is_file():
        return [direct]
    candidates = sorted(path.glob("**/pdm_results.csv"))
    if not candidates:
        raise FileNotFoundError(f"No pdm_results.csv found below {path}")
    return candidates


def load_method_metrics(name: str, path: Path) -> pd.DataFrame:
    frames = [
        pd.read_csv(csv_path, dtype={"token": str}) for csv_path in _metric_csvs(path)
    ]
    frame = pd.concat(frames, ignore_index=True)
    required = {"token", "valid", *METRIC_COLUMNS.values()}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} metrics are missing columns: {sorted(missing)}")
    frame = frame[["token", "valid", *METRIC_COLUMNS.values()]].copy()
    frame["token"] = frame["token"].astype(str).str.lower()
    frame = frame[frame["token"].str.fullmatch(r"[0-9a-f]{16}", na=False)]
    frame["valid"] = _as_valid(frame["valid"])
    for source in METRIC_COLUMNS.values():
        frame[source] = pd.to_numeric(frame[source], errors="coerce")
    frame = frame[
        frame["valid"] & np.isfinite(frame[list(METRIC_COLUMNS.values())]).all(axis=1)
    ].copy()
    if frame["token"].duplicated().any():
        duplicates = frame.loc[frame["token"].duplicated(), "token"].head(5).tolist()
        raise ValueError(f"{name} has duplicate valid tokens: {duplicates}")
    return frame.rename(
        columns={source: metric for metric, source in METRIC_COLUMNS.items()}
    )[["token", *METRIC_COLUMNS]]


def _trajectory_index(path: Path) -> dict[str, Path]:
    root = path.expanduser().resolve()
    token_roots = [root / "tokens"] if (root / "tokens").is_dir() else []
    token_roots.extend(
        candidate / "tokens"
        for candidate in sorted(root.glob("shard_*"))
        if (candidate / "tokens").is_dir()
    )
    if not token_roots:
        raise FileNotFoundError(
            f"No pdm_results.csv or token trajectories found below {root}"
        )
    index: dict[str, Path] = {}
    for token_root in token_roots:
        for token_dir in token_root.iterdir():
            token = token_dir.name.lower()
            trajectory_path = token_dir / "trajectory.npz"
            if (
                not token_dir.is_dir()
                or not re.fullmatch(r"[0-9a-f]{16}", token)
                or not trajectory_path.is_file()
            ):
                continue
            if token in index:
                raise ValueError(f"Duplicate trajectory token below {root}: {token}")
            index[token] = trajectory_path
    if not index:
        raise FileNotFoundError(f"No trajectory.npz files found below {root}")
    return index


def _official_score(cache: Any, prediction: np.ndarray) -> dict[str, float]:
    model_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.5)
    proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    result = pdm_score(
        cache,
        Trajectory(
            poses=np.asarray(prediction, dtype=np.float32),
            trajectory_sampling=model_sampling,
        ),
        proposal_sampling,
        PDMSimulator(proposal_sampling),
        PDMScorer(proposal_sampling, PDMScorerConfig()),
    )
    return {
        "NC": float(result.no_at_fault_collisions),
        "DAC": float(result.drivable_area_compliance),
        "TTC": float(result.time_to_collision_within_bound),
        "EP": float(result.ego_progress),
        "PDMS": float(result.score),
    }


def _safe_rescore_trajectory(
    task: tuple[str, str, str],
) -> dict[str, object]:
    token, raw_trajectory_path, raw_cache_path = task
    try:
        with np.load(raw_trajectory_path, allow_pickle=False) as payload:
            prediction = np.asarray(payload["prediction"], dtype=np.float64)
        if prediction.shape != (8, 3):
            raise ValueError(f"Unexpected prediction shape: {prediction.shape}")
        if not np.isfinite(prediction).all():
            raise ValueError("Prediction contains non-finite values")
        with lzma.open(raw_cache_path, "rb") as stream:
            cache = pickle.load(stream)
        return {
            "ok": True,
            "token": token,
            "result": {"token": token, **_official_score(cache, prediction)},
        }
    except Exception as exc:
        return {"ok": False, "token": token, "error": repr(exc)}


def _rescore_method(
    name: str,
    trajectory_index: dict[str, Path],
    tokens: Iterable[str],
    *,
    cache_root: Path,
    workers: int,
    progress_interval: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_loader = MetricCacheLoader(cache_root.expanduser().resolve())
    token_list = list(tokens)
    tasks = [
        (
            token,
            str(trajectory_index[token]),
            str(cache_loader.metric_cache_paths[token]),
        )
        for token in token_list
        if token in trajectory_index and token in cache_loader.metric_cache_paths
    ]
    missing = sorted(
        set(token_list)
        - set(trajectory_index).intersection(cache_loader.metric_cache_paths)
    )
    failures: list[dict[str, str]] = [
        {"method": name, "token": token, "error": "missing trajectory or cache"}
        for token in missing
    ]
    rows: list[dict[str, object]] = []
    print(f"Rescoring {len(tasks)} {name} trajectories with {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, outcome in enumerate(
            executor.map(_safe_rescore_trajectory, tasks), start=1
        ):
            if bool(outcome["ok"]):
                rows.append(dict(outcome["result"]))
            else:
                failures.append(
                    {
                        "method": name,
                        "token": str(outcome["token"]),
                        "error": str(outcome["error"]),
                    }
                )
            if index % progress_interval == 0 or index == len(tasks):
                print(f"{name} PDM rescoring: {index}/{len(tasks)}")
    return pd.DataFrame(rows), pd.DataFrame(
        failures, columns=["method", "token", "error"]
    )


def _write_rescored_metrics(frame: pd.DataFrame, path: Path) -> None:
    output = frame.rename(
        columns={metric: source for metric, source in METRIC_COLUMNS.items()}
    ).copy()
    output.insert(1, "valid", True)
    output.to_csv(path, index=False)


def _common_tokens(method_frames: dict[str, pd.DataFrame]) -> list[str]:
    token_sets = [set(frame["token"]) for frame in method_frames.values()]
    return sorted(set.intersection(*token_sets))


def _extract_interactions(
    tokens: Iterable[str],
    *,
    cache_root: Path,
    thresholds: InteractionThresholds,
    workers: int,
    progress_interval: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_loader = MetricCacheLoader(cache_root.expanduser().resolve())
    token_list = list(tokens)
    tasks = [
        (token, str(cache_loader.metric_cache_paths[token]), thresholds)
        for token in token_list
        if token in cache_loader.metric_cache_paths
    ]
    failures: list[dict[str, str]] = [
        {"token": token, "stage": "metric_cache", "error": "missing metric cache"}
        for token in sorted(set(token_list) - set(cache_loader.metric_cache_paths))
    ]
    rows: list[dict[str, object]] = []
    print(f"Analyzing {len(tasks)} metric caches with {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, outcome in enumerate(
            executor.map(safe_analyze_metric_cache, tasks), start=1
        ):
            if bool(outcome["ok"]):
                rows.append(dict(outcome["result"]))
            else:
                failures.append(
                    {
                        "token": str(outcome["token"]),
                        "stage": "interaction_geometry",
                        "error": str(outcome["error"]),
                    }
                )
            if index % progress_interval == 0 or index == len(tasks):
                print(f"Interaction geometry: {index}/{len(tasks)}")
    return pd.DataFrame(rows), pd.DataFrame(
        failures, columns=["token", "stage", "error"]
    )


def _load_interaction_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path.expanduser().resolve(), dtype={"token": str})
    required = {
        "token",
        "interacting_vehicle_count",
        "interaction_intensity_v2",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Interaction CSV is missing columns: {sorted(missing)}")
    frame["token"] = frame["token"].astype(str).str.lower()
    frame["interaction_intensity_v2"] = pd.to_numeric(
        frame["interaction_intensity_v2"], errors="coerce"
    )
    frame["interacting_vehicle_count"] = pd.to_numeric(
        frame["interacting_vehicle_count"], errors="coerce"
    )
    frame = frame[
        np.isfinite(frame["interaction_intensity_v2"])
        & np.isfinite(frame["interacting_vehicle_count"])
    ].copy()
    if frame["token"].duplicated().any():
        raise ValueError("Interaction CSV contains duplicate tokens")
    return frame


def _fit_boundaries(frame: pd.DataFrame, source: str) -> dict[str, object]:
    values = frame.loc[
        (frame["interacting_vehicle_count"] > 0)
        & (frame["interaction_intensity_v2"] > 0),
        "interaction_intensity_v2",
    ].to_numpy(dtype=np.float64)
    if len(values) < 4:
        raise ValueError("At least four interacting tokens are required to fit bins")
    weak_max, medium_max, strong_max = np.quantile(values, [1 / 3, 2 / 3, 0.9])
    return {
        "interaction_definition_version": INTERACTION_DEFINITION_VERSION,
        "source": source,
        "interacting_token_count": int(len(values)),
        "weak_max": float(weak_max),
        "medium_max": float(medium_max),
        "strong_max": float(strong_max),
    }


def _load_boundaries(path: Path) -> dict[str, object]:
    payload = json.loads(path.expanduser().resolve().read_text())
    required = {"weak_max", "medium_max", "strong_max"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Bin boundary JSON is missing: {sorted(missing)}")
    limits = [float(payload[key]) for key in ("weak_max", "medium_max", "strong_max")]
    if not all(math.isfinite(value) for value in limits) or limits != sorted(limits):
        raise ValueError("Bin boundaries must be finite and nondecreasing")
    return payload


def assign_interaction_bins(
    frame: pd.DataFrame, boundaries: dict[str, object]
) -> pd.Series:
    weak_max = float(boundaries["weak_max"])
    medium_max = float(boundaries["medium_max"])
    strong_max = float(boundaries["strong_max"])

    def classify(row: pd.Series) -> str:
        if int(row["interacting_vehicle_count"]) <= 0:
            return "none"
        value = float(row["interaction_intensity_v2"])
        if value <= weak_max:
            return "weak"
        if value <= medium_max:
            return "medium"
        if value <= strong_max:
            return "strong"
        return "critical"

    return frame.apply(classify, axis=1)


def _bootstrap_mean_ci(
    values: np.ndarray, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    if len(values) == 0:
        return math.nan, math.nan
    if len(values) == 1 or samples == 0:
        value = float(values[0])
        return value, value
    chunk_size = max(1, min(samples, 256))
    means: list[np.ndarray] = []
    remaining = samples
    while remaining > 0:
        count = min(chunk_size, remaining)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means.append(values[indices].mean(axis=1))
        remaining -= count
    bootstrap_means = np.concatenate(means)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def _performance_summary(
    frame: pd.DataFrame,
    method_names: list[str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for interaction_bin in BIN_ORDER:
        subset = frame[frame["interaction_bin"] == interaction_bin]
        for method in method_names:
            for metric in METRIC_COLUMNS:
                values = subset[f"{method}__{metric}"].to_numpy(dtype=np.float64)
                low, high = _bootstrap_mean_ci(values, bootstrap_samples, rng)
                rows.append(
                    {
                        "interaction_bin": interaction_bin,
                        "method": method,
                        "metric": metric,
                        "token_count": len(values),
                        "mean": float(values.mean()) if len(values) else math.nan,
                        "median": float(np.median(values)) if len(values) else math.nan,
                        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return pd.DataFrame(rows)


def _pairwise_gains(
    frame: pd.DataFrame,
    proposed: str,
    baselines: list[str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    rows: list[dict[str, object]] = []
    for interaction_bin in BIN_ORDER:
        subset = frame[frame["interaction_bin"] == interaction_bin]
        for baseline in baselines:
            for metric in METRIC_COLUMNS:
                delta = (
                    subset[f"{proposed}__{metric}"] - subset[f"{baseline}__{metric}"]
                ).to_numpy(dtype=np.float64)
                low, high = _bootstrap_mean_ci(delta, bootstrap_samples, rng)
                rows.append(
                    {
                        "interaction_bin": interaction_bin,
                        "proposed_method": proposed,
                        "baseline_method": baseline,
                        "metric": metric,
                        "token_count": len(delta),
                        "mean_delta": float(delta.mean()) if len(delta) else math.nan,
                        "median_delta": float(np.median(delta))
                        if len(delta)
                        else math.nan,
                        "win_rate": float(np.mean(delta > 0))
                        if len(delta)
                        else math.nan,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return pd.DataFrame(rows)


def _failure_rates(frame: pd.DataFrame, method_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for interaction_bin in BIN_ORDER:
        subset = frame[frame["interaction_bin"] == interaction_bin]
        for method in method_names:
            count = len(subset)
            rows.append(
                {
                    "interaction_bin": interaction_bin,
                    "method": method,
                    "token_count": count,
                    "nc_zero_rate": float(np.mean(subset[f"{method}__NC"] <= 0.0))
                    if count
                    else math.nan,
                    "dac_zero_rate": float(np.mean(subset[f"{method}__DAC"] <= 0.0))
                    if count
                    else math.nan,
                    "ttc_failure_rate": float(np.mean(subset[f"{method}__TTC"] < 1.0))
                    if count
                    else math.nan,
                    "pdms_below_0p8_rate": float(
                        np.mean(subset[f"{method}__PDMS"] < 0.8)
                    )
                    if count
                    else math.nan,
                }
            )
    return pd.DataFrame(rows)


def _correlations(
    frame: pd.DataFrame, proposed: str, baselines: list[str]
) -> pd.DataFrame:
    def safe_spearman(left: pd.Series, right: pd.Series) -> float:
        valid = np.isfinite(left) & np.isfinite(right)
        left_valid = left[valid]
        right_valid = right[valid]
        if len(left_valid) < 2 or left_valid.nunique() < 2 or right_valid.nunique() < 2:
            return math.nan
        return float(left_valid.corr(right_valid, method="spearman"))

    interacting = frame[frame["interacting_vehicle_count"] > 0]
    intensity = interacting["interaction_intensity_v2"]
    rows: list[dict[str, object]] = []
    for metric in METRIC_COLUMNS:
        rows.append(
            {
                "quantity": f"{proposed}__{metric}",
                "metric": metric,
                "baseline_method": "",
                "token_count": len(interacting),
                "spearman_rho": safe_spearman(
                    intensity, interacting[f"{proposed}__{metric}"]
                ),
            }
        )
        for baseline in baselines:
            delta = (
                interacting[f"{proposed}__{metric}"]
                - interacting[f"{baseline}__{metric}"]
            )
            rows.append(
                {
                    "quantity": f"{proposed}_minus_{baseline}__{metric}",
                    "metric": metric,
                    "baseline_method": baseline,
                    "token_count": len(interacting),
                    "spearman_rho": safe_spearman(intensity, delta),
                }
            )
    return pd.DataFrame(rows)


def _scene_summary(frame: pd.DataFrame, method_names: list[str]) -> pd.DataFrame:
    if "scene_type" not in frame:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (scene_type, interaction_bin), subset in frame.groupby(
        ["scene_type", "interaction_bin"], observed=True
    ):
        for method in method_names:
            row: dict[str, object] = {
                "scene_type": scene_type,
                "interaction_bin": interaction_bin,
                "method": method,
                "token_count": len(subset),
            }
            for metric in METRIC_COLUMNS:
                row[metric] = float(subset[f"{method}__{metric}"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def _write_report(
    output_dir: Path,
    frame: pd.DataFrame,
    performance: pd.DataFrame,
    gains: pd.DataFrame,
    proposed: str,
) -> None:
    def markdown_table(table: pd.DataFrame, float_columns: set[str]) -> list[str]:
        columns = list(table.columns)
        rows = [
            "| " + " | ".join(columns) + " |",
            "|" + "|".join("---" for _ in columns) + "|",
        ]
        for values in table.itertuples(index=False, name=None):
            formatted = []
            for column, value in zip(columns, values):
                if column in float_columns and pd.notna(value):
                    formatted.append(f"{float(value):.6f}")
                else:
                    formatted.append(str(value))
            rows.append("| " + " | ".join(formatted) + " |")
        return rows

    counts = frame["interaction_bin"].value_counts().reindex(BIN_ORDER, fill_value=0)
    lines = [
        "# NAVTEST Performance by Interaction Intensity",
        "",
        f"Proposed method: `{proposed}`",
        f"Common valid tokens: {len(frame)}",
        f"Interaction definition: v{INTERACTION_DEFINITION_VERSION}",
        "",
        "## Token Counts",
        "",
        "| Bin | Tokens |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {int(counts[name])} |" for name in BIN_ORDER)
    lines.extend(["", "## Mean Metrics", ""])
    mean_table = performance.pivot_table(
        index=["interaction_bin", "method"], columns="metric", values="mean"
    ).reset_index()
    lines.extend(markdown_table(mean_table, set(METRIC_COLUMNS)))
    lines.extend(["", "## Proposed Method PDMS Gains", ""])
    pdms_gains = gains[gains["metric"] == "PDMS"]
    lines.extend(
        markdown_table(
            pdms_gains[
                [
                    "interaction_bin",
                    "baseline_method",
                    "token_count",
                    "mean_delta",
                    "ci95_low",
                    "ci95_high",
                    "win_rate",
                ]
            ],
            {"mean_delta", "ci95_low", "ci95_high", "win_rate"},
        )
    )
    (output_dir / "interaction_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _plot_results(
    output_dir: Path,
    frame: pd.DataFrame,
    performance: pd.DataFrame,
    gains: pd.DataFrame,
    proposed: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(dict.fromkeys(performance["method"]))
    x = np.arange(len(BIN_ORDER))
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, metric in zip(axes.flat, METRIC_COLUMNS):
        metric_rows = performance[performance["metric"] == metric]
        for method in methods:
            values = (
                metric_rows[metric_rows["method"] == method]
                .set_index("interaction_bin")
                .reindex(BIN_ORDER)["mean"]
            )
            axis.plot(x, values, marker="o", label=method)
        axis.set_title(metric)
        axis.set_xticks(x, BIN_ORDER, rotation=25)
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower right")
    figure.savefig(output_dir / "metrics_by_interaction_bin.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    pdms = gains[gains["metric"] == "PDMS"]
    for baseline in pdms["baseline_method"].unique():
        values = (
            pdms[pdms["baseline_method"] == baseline]
            .set_index("interaction_bin")
            .reindex(BIN_ORDER)["mean_delta"]
        )
        axis.plot(x, values, marker="o", label=f"{proposed} - {baseline}")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, BIN_ORDER)
    axis.set_ylabel("Paired PDMS gain")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_dir / "paired_pdms_gain_by_interaction_bin.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    interacting = frame.loc[
        frame["interacting_vehicle_count"] > 0, "interaction_intensity_v2"
    ]
    axis.hist(interacting, bins=30, color="#4472c4", edgecolor="white")
    axis.set_xlabel("Interaction intensity v2")
    axis.set_ylabel("Token count")
    figure.savefig(output_dir / "interaction_intensity_distribution.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    methods = dict(args.method)
    if len(methods) != len(args.method):
        raise ValueError("Method names must be unique")
    if args.proposed_method not in methods:
        raise ValueError("--proposed-method must match one --method")
    if args.workers <= 0 or args.progress_interval <= 0:
        raise ValueError("--workers and --progress-interval must be positive")
    if args.rescore_workers is not None and args.rescore_workers <= 0:
        raise ValueError("--rescore-workers must be positive")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples cannot be negative")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.bin_boundaries_json and args.fit_bin_interaction_csv:
        raise ValueError(
            "Use either --bin-boundaries-json or --fit-bin-interaction-csv"
        )

    thresholds = InteractionThresholds(
        horizon_s=args.horizon_s,
        sample_interval_s=args.sample_interval_s,
        nearby_radius_m=args.nearby_radius_m,
        min_agent_displacement_m=args.min_agent_displacement_m,
        min_agent_peak_speed_mps=args.min_agent_peak_speed_mps,
        max_path_conflict_distance_m=args.max_path_conflict_distance_m,
        path_conflict_tolerance_m=args.path_conflict_tolerance_m,
        max_conflict_time_gap_s=args.max_conflict_time_gap_s,
        max_synchronized_distance_m=args.max_synchronized_distance_m,
    )
    thresholds.validate()

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    method_frames: dict[str, pd.DataFrame] = {}
    trajectory_indexes: dict[str, dict[str, Path]] = {}
    token_sets: list[set[str]] = []
    for name, path in methods.items():
        try:
            method_frame = load_method_metrics(name, path)
            method_frames[name] = method_frame
            token_sets.append(set(method_frame["token"]))
        except FileNotFoundError:
            if not args.rescore_missing_methods:
                raise
            trajectory_index = _trajectory_index(path)
            trajectory_indexes[name] = trajectory_index
            token_sets.append(set(trajectory_index))

    common_tokens = sorted(set.intersection(*token_sets))
    if args.max_tokens is not None:
        common_tokens = common_tokens[: args.max_tokens]
    if not common_tokens:
        raise ValueError("Methods have no common valid NAVTEST tokens")

    scoring_failure_frames: list[pd.DataFrame] = []
    for name, trajectory_index in trajectory_indexes.items():
        method_frame, scoring_failures = _rescore_method(
            name,
            trajectory_index,
            common_tokens,
            cache_root=args.metric_cache_path,
            workers=args.rescore_workers or args.workers,
            progress_interval=args.progress_interval,
        )
        method_frames[name] = method_frame
        scoring_failure_frames.append(scoring_failures)
        _write_rescored_metrics(
            method_frame, output_dir / f"{name}_rescored_pdm_results.csv"
        )
    scoring_failures = (
        pd.concat(scoring_failure_frames, ignore_index=True)
        if scoring_failure_frames
        else pd.DataFrame(columns=["method", "token", "error"])
    )
    scoring_failures.to_csv(output_dir / "method_scoring_failures.csv", index=False)

    common_tokens = _common_tokens(method_frames)
    if not common_tokens:
        raise ValueError("No common valid tokens remain after trajectory rescoring")

    if args.interaction_metrics_csv:
        interaction = _load_interaction_csv(args.interaction_metrics_csv)
        failures = pd.DataFrame(columns=["token", "stage", "error"])
    else:
        interaction, failures = _extract_interactions(
            common_tokens,
            cache_root=args.metric_cache_path,
            thresholds=thresholds,
            workers=args.workers,
            progress_interval=args.progress_interval,
        )
    interaction = interaction[
        interaction["token"].isin(set(common_tokens))
    ].sort_values("token")
    interaction.to_csv(output_dir / "interaction_metrics.csv", index=False)
    failures.to_csv(output_dir / "interaction_failures.csv", index=False)

    if args.bin_boundaries_json:
        boundaries = _load_boundaries(args.bin_boundaries_json)
    elif args.fit_bin_interaction_csv:
        fit_frame = _load_interaction_csv(args.fit_bin_interaction_csv)
        boundaries = _fit_boundaries(
            fit_frame, str(args.fit_bin_interaction_csv.expanduser().resolve())
        )
    else:
        boundaries = _fit_boundaries(interaction, "current_common_navtest_tokens")
    (output_dir / "interaction_bin_boundaries.json").write_text(
        json.dumps(boundaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    interaction["interaction_bin"] = assign_interaction_bins(interaction, boundaries)
    interaction["interaction_bin"] = pd.Categorical(
        interaction["interaction_bin"], categories=BIN_ORDER, ordered=True
    )

    combined = interaction.copy()
    for name, method_frame in method_frames.items():
        renamed = method_frame.rename(
            columns={metric: f"{name}__{metric}" for metric in METRIC_COLUMNS}
        )
        combined = combined.merge(
            renamed, on="token", how="inner", validate="one_to_one"
        )
    if args.scene_types_csv and args.scene_types_csv.expanduser().is_file():
        scenes = pd.read_csv(
            args.scene_types_csv.expanduser().resolve(), dtype={"token": str}
        )
        scenes["token"] = scenes["token"].astype(str).str.lower()
        keep = [
            column
            for column in (
                "token",
                "scene_type",
                "motion_type",
                "cumulative_heading_change_deg",
                "intersection_frame_count",
            )
            if column in scenes
        ]
        combined = combined.merge(
            scenes[keep].drop_duplicates("token"),
            on="token",
            how="left",
            validate="one_to_one",
        )
    combined.to_csv(output_dir / "interaction_token_metrics.csv", index=False)

    method_names = list(methods)
    baselines = [name for name in method_names if name != args.proposed_method]
    performance = _performance_summary(
        combined,
        method_names,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    gains = _pairwise_gains(
        combined,
        args.proposed_method,
        baselines,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    failures_by_bin = _failure_rates(combined, method_names)
    correlations = _correlations(combined, args.proposed_method, baselines)
    scene_summary = _scene_summary(combined, method_names)
    performance.to_csv(output_dir / "interaction_performance_summary.csv", index=False)
    gains.to_csv(output_dir / "interaction_pairwise_gains.csv", index=False)
    failures_by_bin.to_csv(output_dir / "interaction_failure_rates.csv", index=False)
    correlations.to_csv(output_dir / "interaction_correlations.csv", index=False)
    scene_summary.to_csv(output_dir / "interaction_scene_type_summary.csv", index=False)
    _write_report(output_dir, combined, performance, gains, args.proposed_method)
    if args.plots:
        _plot_results(output_dir, combined, performance, gains, args.proposed_method)

    summary = {
        "interaction_definition_version": INTERACTION_DEFINITION_VERSION,
        "methods": {
            name: str(path.expanduser().resolve()) for name, path in methods.items()
        },
        "proposed_method": args.proposed_method,
        "common_valid_token_count_before_cache": len(common_tokens),
        "analyzed_token_count": len(combined),
        "failed_interaction_token_count": len(failures),
        "failed_method_scoring_token_count": len(scoring_failures),
        "rescored_methods": sorted(trajectory_indexes),
        "bin_counts": {
            name: int((combined["interaction_bin"] == name).sum()) for name in BIN_ORDER
        },
        "thresholds": asdict(thresholds),
        "boundaries": boundaries,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    (output_dir / "interaction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Analyzed {len(combined)} common valid tokens")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
