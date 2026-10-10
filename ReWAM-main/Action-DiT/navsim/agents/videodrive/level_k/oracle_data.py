"""Validation for causal surrounding-agent inference inputs."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch


ORACLE_AGENT_INPUT_KEYS: Tuple[str, ...] = (
    "agent_current_states",
    "agent_types",
    "agent_mask",
)

LEVEL_ZERO_REFERENCE_INPUT_KEYS: Tuple[str, ...] = (
    "level_zero_ego_reference",
    "level_zero_ego_reference_mask",
    "level_zero_other_references",
    "level_zero_other_reference_mask",
)


def validate_oracle_agent_inputs(
    values: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    """Validate current-frame model inputs without requiring future labels."""

    missing = [key for key in ORACLE_AGENT_INPUT_KEYS if key not in values]
    if missing:
        raise KeyError(
            "Incomplete Oracle-agent model inputs; missing fields: "
            + ", ".join(missing)
        )
    tensors = {
        key: torch.as_tensor(values[key])
        for key in ORACLE_AGENT_INPUT_KEYS
    }
    current_states = tensors["agent_current_states"]
    agent_types = tensors["agent_types"]
    agent_mask = tensors["agent_mask"]

    if current_states.ndim != 2 or current_states.shape[-1] != 8:
        raise ValueError("agent_current_states must have shape [N, 8].")
    num_agents = current_states.shape[0]
    if agent_types.shape != (num_agents,):
        raise ValueError("agent_types must have shape [N].")
    if agent_mask.shape != (num_agents,):
        raise ValueError("agent_mask must have shape [N].")
    if not torch.is_floating_point(current_states):
        raise TypeError("agent_current_states must use a floating dtype.")
    if agent_types.dtype == torch.bool or torch.is_floating_point(agent_types):
        raise TypeError("agent_types must use an integer dtype.")
    if agent_mask.dtype != torch.bool:
        raise TypeError("agent_mask must use bool.")
    if not torch.isfinite(current_states).all():
        raise ValueError("agent_current_states contains non-finite values.")

    present_references = [
        key for key in LEVEL_ZERO_REFERENCE_INPUT_KEYS if key in values
    ]
    if present_references:
        missing_references = [
            key
            for key in LEVEL_ZERO_REFERENCE_INPUT_KEYS
            if key not in values
        ]
        if missing_references:
            raise KeyError(
                "Incomplete Level-0 map references; missing fields: "
                + ", ".join(missing_references)
            )
        reference_tensors = {
            key: torch.as_tensor(values[key])
            for key in LEVEL_ZERO_REFERENCE_INPUT_KEYS
        }
        ego_reference = reference_tensors["level_zero_ego_reference"]
        ego_reference_mask = reference_tensors[
            "level_zero_ego_reference_mask"
        ]
        other_references = reference_tensors[
            "level_zero_other_references"
        ]
        other_reference_mask = reference_tensors[
            "level_zero_other_reference_mask"
        ]
        if ego_reference.ndim != 2 or ego_reference.shape[-1] != 4:
            raise ValueError(
                "level_zero_ego_reference must have shape [T, 4]."
            )
        horizon_steps = ego_reference.shape[0]
        if other_references.shape != (num_agents, horizon_steps, 4):
            raise ValueError(
                "level_zero_other_references must have shape [N, T, 4]."
            )
        if ego_reference_mask.shape != ():
            raise ValueError(
                "level_zero_ego_reference_mask must be a scalar."
            )
        if other_reference_mask.shape != (num_agents,):
            raise ValueError(
                "level_zero_other_reference_mask must have shape [N]."
            )
        if not torch.is_floating_point(ego_reference) or not torch.is_floating_point(
            other_references
        ):
            raise TypeError("Level-0 trajectory references must use a floating dtype.")
        if (
            ego_reference_mask.dtype != torch.bool
            or other_reference_mask.dtype != torch.bool
        ):
            raise TypeError("Level-0 reference masks must use bool.")
        if not torch.isfinite(ego_reference).all() or not torch.isfinite(
            other_references
        ).all():
            raise ValueError("Level-0 trajectory references contain non-finite values.")
        if torch.any(other_reference_mask & ~agent_mask):
            raise ValueError(
                "Padded agents cannot have valid Level-0 map references."
            )
        tensors.update(reference_tensors)
    return tensors



__all__ = [
    "LEVEL_ZERO_REFERENCE_INPUT_KEYS",
    "ORACLE_AGENT_INPUT_KEYS",
    "validate_oracle_agent_inputs",
]
