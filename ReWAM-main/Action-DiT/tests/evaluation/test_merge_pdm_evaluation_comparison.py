from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


METRIC_COLUMNS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)


def _load_module():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "evaluation" / "merge_pdm_evaluation_comparison.py"
    spec = importlib.util.spec_from_file_location("merge_pdm_comparison", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_rank_csv(
    directory: Path,
    *,
    valid: list[bool],
    scores: list[float],
    filename: str = "result.rank_0.csv",
) -> None:
    directory.mkdir(parents=True)
    rows = []
    for index, (is_valid, score) in enumerate(zip(valid, scores, strict=True)):
        row = {"token": f"token_{index}", "valid": is_valid}
        for column in METRIC_COLUMNS:
            row[column] = score if is_valid else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(directory / filename, index=False)


def _run_merge(
    module,
    monkeypatch,
    *,
    baseline_dir: Path,
    finetuned_dir: Path,
    output_dir: Path,
    require_matching_validity_mask: bool = True,
) -> None:
    argv = [
        "merge_pdm_evaluation_comparison.py",
        "--baseline-dir",
        str(baseline_dir),
        "--finetuned-dir",
        str(finetuned_dir),
        "--output-dir",
        str(output_dir),
        "--expected-token-count",
        "4",
        "--bootstrap-resamples",
        "200",
        "--bootstrap-confidence-level",
        "0.95",
        "--bootstrap-seed",
        "7",
    ]
    if require_matching_validity_mask:
        argv.append("--require-matching-validity-mask")
    monkeypatch.setattr(sys, "argv", argv)
    module.main()


def test_merge_writes_paired_confidence_and_tail_analysis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    baseline_dir = tmp_path / "baseline"
    finetuned_dir = tmp_path / "finetuned"
    output_dir = tmp_path / "output"
    valid = [True, True, True, False]
    _write_rank_csv(
        baseline_dir,
        valid=valid,
        scores=[0.9, 0.6, 0.0, np.nan],
    )
    _write_rank_csv(
        finetuned_dir,
        valid=valid,
        scores=[0.8, 0.7, 0.2, np.nan],
    )

    _run_merge(
        module,
        monkeypatch,
        baseline_dir=baseline_dir,
        finetuned_dir=finetuned_dir,
        output_dir=output_dir,
    )

    summary = pd.read_csv(output_dir / "summary.csv").set_index("metric")
    pdms = summary.loc["PDMS"]
    assert pdms["delta"] == pytest.approx(((-0.1) + 0.1 + 0.2) / 3)
    assert int(pdms["improved_count"]) == 2
    assert int(pdms["regressed_count"]) == 1
    assert np.isfinite(pdms["ci_lower"])
    assert np.isfinite(pdms["ci_upper"])

    tail = json.loads((output_dir / "pdms_tail_summary.json").read_text())
    assert tail["baseline_positive_count"] == 2
    assert tail["baseline_positive_regression_count"] == 1
    assert tail["baseline_high_count"] == 1
    assert tail["baseline_high_regression_count"] == 1
    assert tail["positive_to_zero_count"] == 0
    assert tail["zero_to_positive_count"] == 1

    metadata = json.loads((output_dir / "comparison_metadata.json").read_text())
    assert metadata["validity_mask_matches"] is True
    assert metadata["bootstrap_resamples"] == 200
    assert "paired CI" in (output_dir / "summary_report.md").read_text()


def test_merge_accepts_one_premerged_baseline_csv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    baseline_dir = tmp_path / "baseline"
    finetuned_dir = tmp_path / "finetuned"
    valid = [True, True, True, False]
    _write_rank_csv(
        baseline_dir,
        valid=valid,
        scores=[0.9, 0.6, 0.0, np.nan],
        filename="2026.08.27.14.20.38.csv",
    )
    _write_rank_csv(
        finetuned_dir,
        valid=valid,
        scores=[0.8, 0.7, 0.2, np.nan],
    )

    _run_merge(
        module,
        monkeypatch,
        baseline_dir=baseline_dir,
        finetuned_dir=finetuned_dir,
        output_dir=tmp_path / "output",
    )

    summary = pd.read_csv(tmp_path / "output" / "summary.csv")
    assert set(summary["metric"]) >= {"TTC", "DAC", "EP", "PDMS"}


def test_merge_ignores_legacy_average_summary_row(tmp_path: Path) -> None:
    module = _load_module()
    result_dir = tmp_path / "baseline"
    valid = [True, True, True, False]
    _write_rank_csv(
        result_dir,
        valid=valid,
        scores=[0.9, 0.6, 0.0, np.nan],
        filename="merged.csv",
    )
    path = result_dir / "merged.csv"
    frame = pd.read_csv(path)
    average = {column: np.nan for column in frame.columns}
    average["token"] = "average"
    average["valid"] = False
    frame = pd.concat([frame, pd.DataFrame([average])], ignore_index=True)
    frame.to_csv(path, index=False)

    loaded = module._load_rank_results(
        result_dir,
        label="baseline",
        expected_token_count=4,
    )

    assert len(loaded) == 4
    assert "average" not in loaded["token"].tolist()



def test_merge_uses_common_valid_tokens_when_candidate_has_invalid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    baseline_dir = tmp_path / "baseline"
    finetuned_dir = tmp_path / "finetuned"
    output_dir = tmp_path / "output"
    _write_rank_csv(
        baseline_dir,
        valid=[True, True, True, True],
        scores=[0.9, 0.8, 0.7, 0.6],
    )
    _write_rank_csv(
        finetuned_dir,
        valid=[True, False, True, True],
        scores=[0.91, np.nan, 0.72, 0.61],
    )

    _run_merge(
        module,
        monkeypatch,
        baseline_dir=baseline_dir,
        finetuned_dir=finetuned_dir,
        output_dir=output_dir,
        require_matching_validity_mask=False,
    )

    metadata = json.loads(
        (output_dir / "comparison_metadata.json").read_text(encoding="utf-8")
    )
    failed = pd.read_csv(output_dir / "failed_tokens.csv")
    assert metadata["validity_mask_matches"] is False
    assert metadata["common_valid_count"] == 3
    assert metadata["failed_comparison_count"] == 1
    assert failed["token"].tolist() == ["token_1"]


def test_merge_rejects_mismatched_validity_masks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    baseline_dir = tmp_path / "baseline"
    finetuned_dir = tmp_path / "finetuned"
    _write_rank_csv(
        baseline_dir,
        valid=[True, True, True, False],
        scores=[0.9, 0.6, 0.0, np.nan],
    )
    _write_rank_csv(
        finetuned_dir,
        valid=[True, False, True, False],
        scores=[0.8, np.nan, 0.2, np.nan],
    )

    with pytest.raises(ValueError, match="validity masks differ"):
        _run_merge(
            module,
            monkeypatch,
            baseline_dir=baseline_dir,
            finetuned_dir=finetuned_dir,
            output_dir=tmp_path / "output",
        )
