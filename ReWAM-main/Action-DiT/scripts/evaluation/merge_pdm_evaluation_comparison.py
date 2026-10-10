#!/usr/bin/env python3
"""Merge and compare distributed NAVSIM PDM evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "NC": "no_at_fault_collisions",
    "DAC": "drivable_area_compliance",
    "EP": "ego_progress",
    "TTC": "time_to_collision_within_bound",
    "Comfort": "comfort",
    "DDC": "driving_direction_compliance",
    "PDMS": "score",
}
EPSILON = 1e-12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--finetuned-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-token-count", type=int, default=None)
    parser.add_argument(
        "--require-matching-validity-mask",
        action="store_true",
        help="Fail unless baseline and finetuned valid flags match per token.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=0)
    parser.add_argument("--bootstrap-confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def _as_bool(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    unknown = ~normalized.isin({"true", "false"})
    if unknown.any():
        values = sorted(normalized[unknown].unique().tolist())
        raise ValueError(f"{name} contains invalid boolean values: {values}")
    return normalized.eq("true")


def _load_rank_results(
    run_dir: Path,
    *,
    label: str,
    expected_token_count: int | None,
) -> pd.DataFrame:
    paths = sorted(run_dir.glob("*.rank_*.csv"))
    if not paths:
        merged_paths = sorted(run_dir.glob("*.csv"))
        if len(merged_paths) != 1:
            raise FileNotFoundError(
                f"Expected rank CSV shards or one merged CSV for {label}: "
                f"{run_dir}; found {len(merged_paths)} merged candidates"
            )
        paths = merged_paths

    frames = [pd.read_csv(path, dtype={"token": str}) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    result = result.loc[:, ~result.columns.str.startswith("Unnamed:")]

    required = {"token", "valid", *METRICS.values()}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"{label} results are missing columns: {sorted(missing)}")

    # Older merged NAVSIM outputs append a report row named "average".  It is
    # not a scenario token and must not participate in count or metric checks.
    summary_rows = result["token"].astype(str).str.strip().str.lower().eq("average")
    result = result.loc[~summary_rows].copy()
    if result["token"].isna().any() or result["token"].eq("").any():
        raise ValueError(f"{label} results contain empty tokens")
    duplicates = result.loc[result["token"].duplicated(keep=False), "token"]
    if not duplicates.empty:
        raise ValueError(
            f"{label} results contain duplicate tokens: "
            f"{sorted(duplicates.unique().tolist())[:10]}"
        )
    if expected_token_count is not None and len(result) != expected_token_count:
        raise ValueError(
            f"{label} token count mismatch: {len(result)} != "
            f"{expected_token_count}"
        )

    result["valid"] = _as_bool(result["valid"], name=f"{label}.valid")
    valid_rows = result["valid"]
    for column in METRICS.values():
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(result.loc[valid_rows, column]).all():
            raise ValueError(
                f"{label} has non-finite {column} values in valid rows"
            )
    return result.sort_values("token").reset_index(drop=True)


def _write_report(
    path: Path,
    *,
    baseline_dir: Path,
    finetuned_dir: Path,
    all_token_count: int,
    baseline_valid_count: int,
    finetuned_valid_count: int,
    common_valid_count: int,
    summary: pd.DataFrame,
    tail_summary: dict[str, int | float],
    bootstrap_resamples: int,
    bootstrap_confidence_level: float,
) -> None:
    lines = [
        "# NAVSIM Navtest Baseline vs Finetuned",
        "",
        f"- Baseline directory: `{baseline_dir}`",
        f"- Finetuned directory: `{finetuned_dir}`",
        f"- Tokens: {all_token_count}",
        f"- Baseline valid: {baseline_valid_count}",
        f"- Finetuned valid: {finetuned_valid_count}",
        f"- Common valid tokens used for comparison: {common_valid_count}",
        f"- Paired bootstrap resamples: {bootstrap_resamples}",
        f"- Bootstrap confidence level: {bootstrap_confidence_level:.3f}",
        "",
        "| metric | baseline | finetuned | delta | paired CI | improved | regressed | regression rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        if np.isfinite(row.ci_lower) and np.isfinite(row.ci_upper):
            confidence_interval = f"[{row.ci_lower:+.6f}, {row.ci_upper:+.6f}]"
        else:
            confidence_interval = "N/A"
        lines.append(
            f"| {row.metric} | {row.baseline:.6f} | "
            f"{row.finetuned:.6f} | {row.delta:+.6f} | "
            f"{confidence_interval} | {row.improved_count} | "
            f"{row.regressed_count} | {row.regression_rate:.2%} |"
        )
    lines.extend(
        [
            "",
            "## PDMS Tail Analysis",
            "",
            f"- Baseline-positive tokens: {tail_summary['baseline_positive_count']}",
            f"- Baseline-positive regressions: {tail_summary['baseline_positive_regression_count']} "
            f"({tail_summary['baseline_positive_regression_rate']:.2%})",
            f"- Baseline PDMS >= 0.8 tokens: {tail_summary['baseline_high_count']}",
            f"- Baseline PDMS >= 0.8 regressions: {tail_summary['baseline_high_regression_count']} "
            f"({tail_summary['baseline_high_regression_rate']:.2%})",
            f"- Positive-to-zero: {tail_summary['positive_to_zero_count']}",
            f"- Zero-to-positive: {tail_summary['zero_to_positive_count']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _paired_bootstrap_intervals(
    deltas: np.ndarray,
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return percentile CIs for paired mean deltas using shared resamples."""

    metric_count = deltas.shape[1]
    if resamples == 0:
        missing = np.full(metric_count, np.nan, dtype=np.float64)
        return missing.copy(), missing

    rng = np.random.default_rng(seed)
    token_count = deltas.shape[0]
    bootstrap_means = np.empty((resamples, metric_count), dtype=np.float64)
    batch_size = 32
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = rng.integers(
            0,
            token_count,
            size=(stop - start, token_count),
            dtype=np.int32,
        )
        bootstrap_means[start:stop] = deltas[indices].mean(axis=1)

    alpha = (1.0 - confidence_level) / 2.0
    lower = np.quantile(bootstrap_means, alpha, axis=0)
    upper = np.quantile(bootstrap_means, 1.0 - alpha, axis=0)
    return lower, upper


def main() -> None:
    args = _parse_args()
    if args.bootstrap_resamples < 0:
        raise ValueError("bootstrap_resamples must be non-negative")
    if not 0.0 < args.bootstrap_confidence_level < 1.0:
        raise ValueError("bootstrap_confidence_level must lie in (0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _load_rank_results(
        args.baseline_dir,
        label="baseline",
        expected_token_count=args.expected_token_count,
    )
    finetuned = _load_rank_results(
        args.finetuned_dir,
        label="finetuned",
        expected_token_count=args.expected_token_count,
    )

    baseline_tokens = set(baseline["token"])
    finetuned_tokens = set(finetuned["token"])
    if baseline_tokens != finetuned_tokens:
        missing_finetuned = sorted(baseline_tokens - finetuned_tokens)
        missing_baseline = sorted(finetuned_tokens - baseline_tokens)
        raise ValueError(
            "Evaluation token sets differ: "
            f"missing_finetuned={missing_finetuned[:10]}, "
            f"missing_baseline={missing_baseline[:10]}"
        )

    baseline.to_csv(args.output_dir / "baseline_all_ranks.csv", index=False)
    finetuned.to_csv(args.output_dir / "finetuned_all_ranks.csv", index=False)

    baseline_indexed = baseline.set_index("token")
    finetuned_indexed = finetuned.set_index("token").loc[baseline_indexed.index]
    validity_mismatch = baseline_indexed["valid"] != finetuned_indexed["valid"]
    validity_mask_matches = not bool(validity_mismatch.any())
    if args.require_matching_validity_mask and not validity_mask_matches:
        mismatch_tokens = baseline_indexed.index[validity_mismatch].tolist()
        raise ValueError(
            "Baseline and finetuned validity masks differ: "
            f"count={len(mismatch_tokens)}, tokens={mismatch_tokens[:20]}"
        )

    comparison = pd.DataFrame(index=baseline_indexed.index)
    comparison.index.name = "token"
    comparison["baseline_valid"] = baseline_indexed["valid"]
    comparison["finetuned_valid"] = finetuned_indexed["valid"]
    comparison["comparison_valid"] = (
        comparison["baseline_valid"] & comparison["finetuned_valid"]
    )

    for metric, column in METRICS.items():
        comparison[f"baseline_{metric}"] = baseline_indexed[column]
        comparison[f"finetuned_{metric}"] = finetuned_indexed[column]
        comparison[f"delta_{metric}"] = (
            comparison[f"finetuned_{metric}"]
            - comparison[f"baseline_{metric}"]
        )

    comparison = comparison.reset_index()
    comparison.to_csv(args.output_dir / "token_comparison.csv", index=False)

    failed = comparison.loc[
        ~comparison["comparison_valid"],
        ["token", "baseline_valid", "finetuned_valid"],
    ].copy()
    failed.to_csv(args.output_dir / "failed_tokens.csv", index=False)

    common_valid = comparison["comparison_valid"]
    common_valid_count = int(common_valid.sum())
    if common_valid_count == 0:
        raise ValueError("No common valid tokens are available for comparison")

    valid_comparison = comparison.loc[common_valid]
    delta_columns = [f"delta_{metric}" for metric in METRICS]
    delta_matrix = valid_comparison[delta_columns].to_numpy(dtype=np.float64)
    ci_lower, ci_upper = _paired_bootstrap_intervals(
        delta_matrix,
        resamples=args.bootstrap_resamples,
        confidence_level=args.bootstrap_confidence_level,
        seed=args.bootstrap_seed,
    )

    summary_rows = []
    for metric_index, metric in enumerate(METRICS):
        baseline_mean = comparison.loc[common_valid, f"baseline_{metric}"].mean()
        finetuned_mean = comparison.loc[common_valid, f"finetuned_{metric}"].mean()
        metric_deltas = delta_matrix[:, metric_index]
        improved_count = int((metric_deltas > EPSILON).sum())
        regressed_count = int((metric_deltas < -EPSILON).sum())
        summary_rows.append(
            {
                "metric": metric,
                "baseline": baseline_mean,
                "finetuned": finetuned_mean,
                "delta": finetuned_mean - baseline_mean,
                "baseline_percent": baseline_mean * 100.0,
                "finetuned_percent": finetuned_mean * 100.0,
                "delta_percentage_points": (
                    finetuned_mean - baseline_mean
                ) * 100.0,
                "ci_lower": ci_lower[metric_index],
                "ci_upper": ci_upper[metric_index],
                "improved_count": improved_count,
                "regressed_count": regressed_count,
                "unchanged_count": (
                    common_valid_count - improved_count - regressed_count
                ),
                "improvement_rate": improved_count / common_valid_count,
                "regression_rate": regressed_count / common_valid_count,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    baseline_pdms = valid_comparison["baseline_PDMS"]
    finetuned_pdms = valid_comparison["finetuned_PDMS"]
    baseline_positive = baseline_pdms > EPSILON
    baseline_high = baseline_pdms >= 0.8 - EPSILON
    pdms_regressed = finetuned_pdms < baseline_pdms - EPSILON
    positive_to_zero = baseline_positive & (finetuned_pdms <= EPSILON)
    zero_to_positive = (baseline_pdms <= EPSILON) & (finetuned_pdms > EPSILON)
    baseline_positive_count = int(baseline_positive.sum())
    baseline_high_count = int(baseline_high.sum())
    baseline_positive_regression_count = int(
        (baseline_positive & pdms_regressed).sum()
    )
    baseline_high_regression_count = int((baseline_high & pdms_regressed).sum())
    tail_summary = {
        "baseline_positive_count": baseline_positive_count,
        "baseline_positive_regression_count": baseline_positive_regression_count,
        "baseline_positive_regression_rate": (
            baseline_positive_regression_count / baseline_positive_count
            if baseline_positive_count
            else 0.0
        ),
        "baseline_high_count": baseline_high_count,
        "baseline_high_regression_count": baseline_high_regression_count,
        "baseline_high_regression_rate": (
            baseline_high_regression_count / baseline_high_count
            if baseline_high_count
            else 0.0
        ),
        "positive_to_zero_count": int(positive_to_zero.sum()),
        "zero_to_positive_count": int(zero_to_positive.sum()),
    }
    (args.output_dir / "pdms_tail_summary.json").write_text(
        json.dumps(tail_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    baseline_valid_count = int(baseline["valid"].sum())
    finetuned_valid_count = int(finetuned["valid"].sum())
    metadata = {
        "baseline_dir": str(args.baseline_dir.resolve()),
        "finetuned_dir": str(args.finetuned_dir.resolve()),
        "all_token_count": len(comparison),
        "baseline_valid_count": baseline_valid_count,
        "finetuned_valid_count": finetuned_valid_count,
        "common_valid_count": common_valid_count,
        "failed_comparison_count": len(failed),
        "aggregation": "mean_over_common_valid_tokens",
        "require_matching_validity_mask": args.require_matching_validity_mask,
        "validity_mask_matches": validity_mask_matches,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_confidence_level": args.bootstrap_confidence_level,
        "bootstrap_seed": args.bootstrap_seed,
    }
    (args.output_dir / "comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(
        args.output_dir / "summary_report.md",
        baseline_dir=args.baseline_dir,
        finetuned_dir=args.finetuned_dir,
        all_token_count=len(comparison),
        baseline_valid_count=baseline_valid_count,
        finetuned_valid_count=finetuned_valid_count,
        common_valid_count=common_valid_count,
        summary=summary,
        tail_summary=tail_summary,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_confidence_level=args.bootstrap_confidence_level,
    )
    print(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
