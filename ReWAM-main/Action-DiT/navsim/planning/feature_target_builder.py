"""Shared feature and target builder interfaces.

Feature builders are part of the inference path. Target builders remain in
this neutral module for compatibility with NAVSIM agents that also support
offline cache generation, without importing the training package.
"""

from abc import abstractmethod
from typing import Dict

from torch import Tensor

from navsim.common.dataclasses import AgentInput, Scene


class AbstractFeatureBuilder:
    """Build causal model inputs from an ``AgentInput``."""

    @abstractmethod
    def get_unique_name(self) -> str:
        """Return the unique cache name for this feature."""

    @abstractmethod
    def compute_features(self, agent_input: AgentInput) -> Dict[str, Tensor]:
        """Build features without access to future annotations."""


class AbstractTargetBuilder:
    """Build supervision targets from an annotated scene."""

    @abstractmethod
    def get_unique_name(self) -> str:
        """Return the unique cache name for this target."""

    @abstractmethod
    def compute_targets(self, scene: Scene) -> Dict[str, Tensor]:
        """Build targets that may use ground-truth annotations."""


__all__ = ["AbstractFeatureBuilder", "AbstractTargetBuilder"]
