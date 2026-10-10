"""Shared Flow-Matching expert for surrounding-agent trajectories."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import CompatibleRMSNorm, MaskedMultiHeadAttention
from .types import AgentStateBatch, StrategyTokenBatch


def select_video_layer_indices(
    *,
    num_expert_layers: int,
    num_video_layers: int,
) -> tuple[int, ...]:
    """Select deterministic, evenly spaced Video blocks for the Other expert."""

    for name, value in (
        ("num_expert_layers", num_expert_layers),
        ("num_video_layers", num_video_layers),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must use an integer value.")
    num_expert_layers = int(num_expert_layers)
    num_video_layers = int(num_video_layers)
    if num_expert_layers <= 0:
        raise ValueError("num_expert_layers must be positive.")
    if num_video_layers <= 0:
        raise ValueError("num_video_layers must be positive.")
    if num_expert_layers > num_video_layers:
        raise ValueError("num_expert_layers cannot exceed num_video_layers.")
    if num_expert_layers == 1:
        return (num_video_layers - 1,)

    indices = torch.linspace(
        0,
        num_video_layers - 1,
        steps=num_expert_layers,
        dtype=torch.float64,
    ).round().to(torch.long)
    result = tuple(int(index) for index in indices.tolist())
    if len(set(result)) != num_expert_layers:
        raise RuntimeError("Even Video-layer selection produced duplicate indices.")
    return result


def _sinusoidal_embedding(values: Tensor, embedding_dim: int) -> Tensor:
    """Create a standard sinusoidal embedding for arbitrary-shaped values."""

    half_dim = embedding_dim // 2
    if half_dim == 0:
        return values[..., None]
    exponent = -math.log(10000.0) * torch.arange(
        half_dim,
        device=values.device,
        dtype=torch.float32,
    ) / max(half_dim - 1, 1)
    frequencies = torch.exp(exponent)
    angles = values.float()[..., None] * frequencies
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    if embedding_dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class _SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        intermediate_dim = hidden_dim * 2
        self.input_projection = nn.Linear(hidden_dim, intermediate_dim * 2)
        self.output_projection = nn.Linear(intermediate_dim, hidden_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate, value = self.input_projection(hidden_states).chunk(2, dim=-1)
        return self.output_projection(F.silu(gate) * value)


class OtherAgentTransformerBlock(nn.Module):
    """One shared temporal/video/strategy block for every surrounding agent."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        video_hidden_dim: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.self_norm = CompatibleRMSNorm(hidden_dim)
        self.video_norm = CompatibleRMSNorm(hidden_dim)
        self.strategy_norm = CompatibleRMSNorm(hidden_dim)
        self.ff_norm = CompatibleRMSNorm(hidden_dim)
        self.self_attention = MaskedMultiHeadAttention(
            query_dim=hidden_dim,
            context_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        self.video_attention = MaskedMultiHeadAttention(
            query_dim=hidden_dim,
            context_dim=video_hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        self.strategy_attention = MaskedMultiHeadAttention(
            query_dim=hidden_dim,
            context_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        self.feed_forward = _SwiGLUFeedForward(hidden_dim)
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_dim) / hidden_dim**0.5
        )

    def forward(
        self,
        *,
        hidden_states: Tensor,
        conditioning: Tensor,
        video_hidden_states: Tensor,
        strategy_tokens: Tensor,
        strategy_visibility: Tensor,
        agent_mask: Tensor,
    ) -> Tensor:
        batch_size, num_agents, num_steps, hidden_dim = hidden_states.shape
        if conditioning.shape != (
            batch_size,
            num_agents,
            num_steps,
            6 * hidden_dim,
        ):
            raise ValueError("conditioning must have shape [B, N, T, 6*D].")

        ada_values = (
            self.scale_shift_table[None, None, None]
            + conditioning.view(
                batch_size, num_agents, num_steps, 6, hidden_dim
            )
        )
        shift_msa, scale_msa, gate_msa, shift_ff, scale_ff, gate_ff = (
            ada_values.unbind(dim=3)
        )

        temporal_input = self.self_norm(hidden_states)
        temporal_input = temporal_input * (1 + scale_msa) + shift_msa
        temporal_input = temporal_input.reshape(
            batch_size * num_agents, num_steps, hidden_dim
        )
        temporal_output = self.self_attention(
            temporal_input,
            temporal_input,
        ).view(batch_size, num_agents, num_steps, hidden_dim)
        hidden_states = hidden_states + temporal_output * gate_msa

        flat_hidden = hidden_states.reshape(
            batch_size, num_agents * num_steps, hidden_dim
        )
        video_output = self.video_attention(
            self.video_norm(flat_hidden),
            video_hidden_states,
        )
        flat_hidden = flat_hidden + video_output

        strategy_output = self.strategy_attention(
            self.strategy_norm(flat_hidden),
            strategy_tokens,
            strategy_visibility,
        )
        flat_hidden = flat_hidden + strategy_output
        hidden_states = flat_hidden.view(
            batch_size, num_agents, num_steps, hidden_dim
        )

        ff_input = self.ff_norm(hidden_states)
        ff_input = ff_input * (1 + scale_ff) + shift_ff
        hidden_states = hidden_states + self.feed_forward(ff_input) * gate_ff
        return hidden_states * agent_mask[..., None, None].to(
            dtype=hidden_states.dtype
        )


class SharedOtherAgentExpert(nn.Module):
    """One parameter-shared expert that predicts every valid surrounding agent."""

    def __init__(
        self,
        *,
        horizon_steps: int = 8,
        input_dim: int = 4,
        output_dim: int = 4,
        hidden_dim: int = 512,
        video_hidden_dim: int = 2048,
        num_heads: int = 16,
        head_dim: int = 32,
        num_layers: int = 8,
        num_video_layers: int = 28,
        state_position_scale: float = 80.0,
        state_velocity_scale: float = 30.0,
        state_size_scale: float = 10.0,
    ) -> None:
        super().__init__()
        integer_values = {
            "horizon_steps": horizon_steps,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "video_hidden_dim": video_hidden_dim,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "num_layers": num_layers,
            "num_video_layers": num_video_layers,
        }
        for name, value in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must use an integer value.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")

        horizon_steps = int(horizon_steps)
        input_dim = int(input_dim)
        output_dim = int(output_dim)
        hidden_dim = int(hidden_dim)
        video_hidden_dim = int(video_hidden_dim)
        num_heads = int(num_heads)
        head_dim = int(head_dim)
        num_layers = int(num_layers)
        num_video_layers = int(num_video_layers)

        scale_values = {
            "state_position_scale": state_position_scale,
            "state_velocity_scale": state_velocity_scale,
            "state_size_scale": state_size_scale,
        }
        for name, value in scale_values.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must use a real numeric value.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")

        state_position_scale = float(state_position_scale)
        state_velocity_scale = float(state_velocity_scale)
        state_size_scale = float(state_size_scale)

        if hidden_dim != num_heads * head_dim:
            raise ValueError("hidden_dim must equal num_heads * head_dim.")

        self.horizon_steps = horizon_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.video_hidden_dim = video_hidden_dim
        self.num_layers = num_layers
        self.num_video_layers = num_video_layers
        self.state_position_scale = state_position_scale
        self.state_velocity_scale = state_velocity_scale
        self.state_size_scale = state_size_scale
        self.video_layer_indices = select_video_layer_indices(
            num_expert_layers=num_layers,
            num_video_layers=num_video_layers,
        )
        self.video_to_expert_index = {
            video_index: expert_index
            for expert_index, video_index in enumerate(self.video_layer_indices)
        }

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.state_film = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 6),
        )
        self.timestep_film = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 6),
        )
        time_positions = _sinusoidal_embedding(
            torch.arange(horizon_steps, dtype=torch.float32),
            hidden_dim,
        )
        self.register_buffer(
            "trajectory_time_embedding",
            time_positions,
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            [
                OtherAgentTransformerBlock(
                    hidden_dim=hidden_dim,
                    video_hidden_dim=video_hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def _expand_timestep(
        self,
        timestep: Tensor | float | int,
        *,
        batch_size: int,
        num_agents: int,
        device: torch.device,
    ) -> Tensor:
        if isinstance(timestep, Real) and not isinstance(timestep, bool):
            values = torch.full(
                (batch_size, num_agents, self.horizon_steps),
                float(timestep),
                device=device,
            )
        elif isinstance(timestep, Tensor):
            values = timestep.to(device=device)
            if values.ndim == 0:
                values = values.expand(
                    batch_size, num_agents, self.horizon_steps
                )
            elif values.shape == (batch_size,):
                values = values[:, None, None].expand(
                    batch_size, num_agents, self.horizon_steps
                )
            elif values.shape == (batch_size, self.horizon_steps):
                values = values[:, None].expand(
                    batch_size, num_agents, self.horizon_steps
                )
            elif values.shape != (
                batch_size,
                num_agents,
                self.horizon_steps,
            ):
                raise ValueError(
                    "timestep must be scalar, [B], [B,T], or [B,N,T]."
                )
        else:
            raise TypeError("timestep must be numeric or a tensor.")
        if not torch.isfinite(values).all():
            raise ValueError("timestep must contain finite values.")
        return values

    def _strategy_visibility(
        self,
        strategy_tokens: StrategyTokenBatch,
        *,
        num_agents: int,
        num_steps: int,
    ) -> Tensor:
        batch_size = strategy_tokens.tokens.shape[0]
        if strategy_tokens.tokens.shape[1] != num_agents + 1:
            raise ValueError("Strategy tokens must contain Ego plus N agents.")
        visible = strategy_tokens.token_mask[:, None, :].expand(
            batch_size, num_agents, num_agents + 1
        ).clone()
        focal_indices = torch.arange(
            num_agents,
            device=visible.device,
        )
        visible[:, focal_indices, focal_indices + 1] = False
        return visible[:, :, None, :].expand(
            batch_size, num_agents, num_steps, num_agents + 1
        ).reshape(batch_size, num_agents * num_steps, num_agents + 1)

    def preprocess(
        self,
        *,
        noisy_trajectories: Tensor,
        timestep: Tensor | float | int,
        agent_states: AgentStateBatch,
        strategy_tokens: StrategyTokenBatch,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Create hidden states, FiLM conditioning, and focal visibility."""

        agent_states.validate()
        strategy_tokens.validate()
        if (
            noisy_trajectories.ndim != 4
            or noisy_trajectories.shape[-2:] != (self.horizon_steps, self.input_dim)
        ):
            raise ValueError(
                "noisy_trajectories must have shape [B, N, horizon_steps, input_dim]."
            )
        if not torch.is_floating_point(noisy_trajectories):
            raise TypeError("noisy_trajectories must use a floating-point dtype.")

        batch_size, num_agents, _, _ = noisy_trajectories.shape
        if agent_states.current_states.shape[:2] != (batch_size, num_agents):
            raise ValueError("noisy_trajectories and agent_states must share B and N.")
        references = (
            agent_states.current_states,
            strategy_tokens.tokens,
        )
        if any(tensor.device != noisy_trajectories.device for tensor in references):
            raise ValueError("Other-Agent Expert inputs must share one device.")
        if any(tensor.dtype != noisy_trajectories.dtype for tensor in references):
            raise TypeError(
                "Floating-point Other-Agent Expert inputs must share one dtype."
            )

        valid_noisy = noisy_trajectories[agent_states.agent_mask]
        if valid_noisy.numel() and not torch.isfinite(valid_noisy).all():
            raise ValueError("Valid-agent noisy trajectories must be finite.")
        safe_noisy = torch.where(
            agent_states.agent_mask[..., None, None],
            noisy_trajectories,
            torch.zeros_like(noisy_trajectories),
        )
        safe_states = torch.where(
            agent_states.agent_mask[..., None],
            agent_states.current_states,
            torch.zeros_like(agent_states.current_states),
        )

        normalized_states = safe_states.clone()
        normalized_states[..., :2] /= self.state_position_scale
        normalized_states[..., 4:6] /= self.state_velocity_scale
        normalized_states[..., 6:8] /= self.state_size_scale
        state_embedding = self.state_encoder(normalized_states)
        state_condition = self.state_film(state_embedding)[:, :, None]

        timestep_values = self._expand_timestep(
            timestep,
            batch_size=batch_size,
            num_agents=num_agents,
            device=noisy_trajectories.device,
        )
        timestep_embedding = _sinusoidal_embedding(
            timestep_values,
            self.hidden_dim,
        ).to(dtype=noisy_trajectories.dtype)
        timestep_condition = self.timestep_film(timestep_embedding)
        conditioning = timestep_condition + state_condition

        position_embedding = self.trajectory_time_embedding.to(
            device=noisy_trajectories.device,
            dtype=noisy_trajectories.dtype,
        )
        hidden_states = (
            self.input_projection(safe_noisy)
            + position_embedding[None, None]
        )
        hidden_states = hidden_states * agent_states.agent_mask[
            ..., None, None
        ].to(dtype=hidden_states.dtype)
        visibility = self._strategy_visibility(
            strategy_tokens,
            num_agents=num_agents,
            num_steps=self.horizon_steps,
        )
        return hidden_states, conditioning, visibility

    def forward_block(
        self,
        *,
        expert_block_index: int,
        hidden_states: Tensor,
        conditioning: Tensor,
        video_hidden_states: Tensor,
        strategy_tokens: StrategyTokenBatch,
        strategy_visibility: Tensor,
        agent_mask: Tensor,
    ) -> Tensor:
        if not 0 <= expert_block_index < self.num_layers:
            raise IndexError("expert_block_index is out of range.")
        if video_hidden_states.ndim != 3:
            raise ValueError("video_hidden_states must have shape [B, L, C].")
        if video_hidden_states.shape[0] != hidden_states.shape[0]:
            raise ValueError("Video and Other-Agent states must share batch size.")
        if video_hidden_states.shape[-1] != self.video_hidden_dim:
            raise ValueError("Unexpected Video hidden dimension.")
        if video_hidden_states.device != hidden_states.device:
            raise ValueError(
                "Video and Other-Agent hidden states must share one device."
            )
        if video_hidden_states.dtype != hidden_states.dtype:
            raise TypeError(
                "Video and Other-Agent hidden states must share one dtype."
            )
        return self.blocks[expert_block_index](
            hidden_states=hidden_states,
            conditioning=conditioning,
            video_hidden_states=video_hidden_states,
            strategy_tokens=strategy_tokens.tokens,
            strategy_visibility=strategy_visibility,
            agent_mask=agent_mask,
        )

    def postprocess(self, hidden_states: Tensor, agent_mask: Tensor) -> Tensor:
        output = self.output_projection(self.output_norm(hidden_states))
        return output * agent_mask[..., None, None].to(dtype=output.dtype)

    def forward(
        self,
        *,
        noisy_trajectories: Tensor,
        timestep: Tensor | float | int,
        agent_states: AgentStateBatch,
        strategy_tokens: StrategyTokenBatch,
        video_hidden_states: Sequence[Tensor],
    ) -> Tensor:
        """Run the standalone expert using full or preselected Video states."""

        hidden_states, conditioning, visibility = self.preprocess(
            noisy_trajectories=noisy_trajectories,
            timestep=timestep,
            agent_states=agent_states,
            strategy_tokens=strategy_tokens,
        )
        if len(video_hidden_states) == self.num_video_layers:
            selected_video_states = [
                video_hidden_states[index] for index in self.video_layer_indices
            ]
        elif len(video_hidden_states) == self.num_layers:
            selected_video_states = list(video_hidden_states)
        else:
            raise ValueError(
                "video_hidden_states must contain every Video layer or one "
                "entry per Other-Agent block."
            )

        for block_index, video_states in enumerate(selected_video_states):
            hidden_states = self.forward_block(
                expert_block_index=block_index,
                hidden_states=hidden_states,
                conditioning=conditioning,
                video_hidden_states=video_states,
                strategy_tokens=strategy_tokens,
                strategy_visibility=visibility,
                agent_mask=agent_states.agent_mask,
            )
        return self.postprocess(hidden_states, agent_states.agent_mask)
