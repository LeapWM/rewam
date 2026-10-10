"""Inference-only metadata for Level-K evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .config import LevelKConfig


ORIGINAL_MODEL_NAME = "DriveLaW (Original Visual Perception)"
ORACLE_LEVEL_K_MODEL_NAME = "DriveLaW-LevelK (Oracle Current-Agent States)"


def build_inference_metadata(config: LevelKConfig) -> dict[str, Any]:
    """Describe the model inputs and reasoning depth used for evaluation."""

    is_oracle_level_k = bool(config.enabled and config.oracle_agent_states)
    if is_oracle_level_k:
        model_name = ORACLE_LEVEL_K_MODEL_NAME
        comparison_family = "oracle_level_k"
        perception_source = "navsim_oracle_current_annotations"
    elif config.enabled:
        model_name = "DriveLaW-LevelK (Detected/Tracked Current-Agent States)"
        comparison_family = "detected_tracked_level_k"
        perception_source = "detector_tracker_current_agent_states"
    else:
        model_name = ORIGINAL_MODEL_NAME
        comparison_family = "original_visual_drivelaw"
        perception_source = "original_visual_perception"

    return {
        "schema": "rewam_inference_metadata",
        "schema_version": 1,
        "model_name": model_name,
        "comparison_family": comparison_family,
        "evaluation_variant": (
            f"{comparison_family}_k{config.reasoning_level}"
            if config.enabled
            else comparison_family
        ),
        "level_k_enabled": bool(config.enabled),
        "reasoning_level": int(config.reasoning_level),
        "max_reasoning_level": int(config.max_reasoning_level),
        "max_agents": int(config.max_agents),
        "agent_range": float(config.agent_range),
        "oracle_agent_states": is_oracle_level_k,
        "perception_source": perception_source,
        "future_annotations_used_as_model_input": False,
        "level_zero_policy": {
            "mode": config.level_zero_mode,
            "fallback": config.level_zero_fallback,
            "future_annotations_used": False,
        },
    }


def write_inference_metadata(
    output_dir: str | Path,
    metadata: Mapping[str, Any],
    *,
    artifact_type: str,
    filename: str = "inference_metadata.json",
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a self-describing metadata sidecar next to an inference artifact."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload["artifact_type"] = artifact_type
    if extra:
        payload["artifact"] = dict(extra)
    output_path = destination / filename
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


__all__ = [
    "ORACLE_LEVEL_K_MODEL_NAME",
    "ORIGINAL_MODEL_NAME",
    "build_inference_metadata",
    "write_inference_metadata",
]
