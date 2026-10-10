"""Inference agent interface used by NAVSIM evaluation."""

from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.feature_target_builder import AbstractFeatureBuilder


class AbstractAgent(torch.nn.Module, ABC):
    """Interface implemented by agents evaluated in NAVSIM."""

    def __init__(self, requires_scene: bool = False):
        super().__init__()
        self.requires_scene = requires_scene

    @abstractmethod
    def name(self) -> str:
        """Return the agent name used in logs and artifacts."""

    @abstractmethod
    def get_sensor_config(self) -> SensorConfig:
        """Return the sensors required to build inference inputs."""

    @abstractmethod
    def initialize(self) -> None:
        """Load model components and checkpoint weights."""

    def forward(
        self, features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Run a model forward pass."""

        raise NotImplementedError

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Return causal feature builders used during inference."""

        raise NotImplementedError

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """Predict one trajectory from causal agent inputs."""

        self.eval()
        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        features = {key: value.unsqueeze(0) for key, value in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["trajectory"].squeeze(0).cpu().numpy()
        return Trajectory(poses)
