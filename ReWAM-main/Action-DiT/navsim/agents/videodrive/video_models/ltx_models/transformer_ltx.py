# Copyright 2025 The Lightricks team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dataclasses import asdict, dataclass, field
from pathlib import Path

from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, deprecate, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
#from diffusers.models import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward, Attention
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
import torch.utils.checkpoint
from navsim.agents.videodrive.video_models.diffusion_planner.diffusion_planner import preprocessing_action_states, add_action_expert, ActionTrajectoryScorer
from navsim.agents.videodrive.level_k.other_agent_expert import SharedOtherAgentExpert
from navsim.agents.videodrive.level_k.level_zero import LevelZeroPolicy
from navsim.agents.videodrive.level_k.strategy_conditioning import (
    EgoStrategyContext,
    prepare_ego_strategy_context,
)
from navsim.agents.videodrive.level_k.strategy_encoder import StrategyTokenEncoder
from navsim.agents.videodrive.level_k.types import (
    AgentStateBatch,
    LevelZeroReferenceBatch,
    StrategyTokenBatch,
    StrategyTrajectoryBatch,
)



logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass(frozen=True)
class ContextParallelInput:
    """
    Configuration for splitting an input tensor across context parallel region.

    Args:
        split_dim (`int`):
            The dimension along which to split the tensor.
        expected_dims (`int`, *optional*):
            The expected number of dimensions of the tensor. If provided, a check will be performed to ensure that the
            tensor has the expected number of dimensions before splitting.
        split_output (`bool`, *optional*, defaults to `False`):
            Whether to split the output tensor of the layer along the given `split_dim` instead of the input tensor.
            This is useful for layers whose outputs should be split after it does some preprocessing on the inputs (ex:
            RoPE).
    """

    split_dim: int
    expected_dims: Optional[int] = None
    split_output: bool = False

    def __repr__(self):
        return f"ContextParallelInput(split_dim={self.split_dim}, expected_dims={self.expected_dims}, split_output={self.split_output})"


@dataclass(frozen=True)
class ContextParallelOutput:
    """
    Configuration for gathering an output tensor across context parallel region.

    Args:
        gather_dim (`int`):
            The dimension along which to gather the tensor.
        expected_dims (`int`, *optional*):
            The expected number of dimensions of the tensor. If provided, a check will be performed to ensure that the
            tensor has the expected number of dimensions before gathering.
    """

    gather_dim: int
    expected_dims: Optional[int] = None

    def __repr__(self):
        return f"ContextParallelOutput(gather_dim={self.gather_dim}, expected_dims={self.expected_dims})"

class LTXVideoAttentionProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`LTXVideoAttentionProcessor2_0` is deprecated and this will be removed in a future version. Please use `LTXVideoAttnProcessor`"
        deprecate("LTXVideoAttentionProcessor2_0", "1.0.0", deprecation_message)

        return LTXVideoAttnProcessor(*args, **kwargs)


class LTXVideoAttnProcessor:
    r"""
    Processor for implementing attention (SDPA is used by default if you're using PyTorch 2.0). This is used in the LTX
    model. It applies a normalization layer and rotary embedding on the query and key vector.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if is_torch_version("<", "2.0"):
            raise ValueError(
                "LTX attention processors require a minimum PyTorch version of 2.0. Please upgrade your PyTorch installation."
            )

    def __call__(
        self,
        attn: "LTXAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            #parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class LTXAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = LTXVideoAttnProcessor
    _available_processors = [LTXVideoAttnProcessor]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        kv_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        qk_norm: str = "rms_norm_across_heads",
        processor=None,
    ):
        super().__init__()
        if qk_norm != "rms_norm_across_heads":
            raise NotImplementedError("Only 'rms_norm_across_heads' is supported as a valid value for `qk_norm`.")

        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = query_dim
        self.heads = heads

        norm_eps = 1e-5
        norm_elementwise_affine = True
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head * kv_heads, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_v = torch.nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(torch.nn.Dropout(dropout))

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)


class LTXVideoRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        base_num_frames: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        patch_size: int = 1,
        patch_size_t: int = 1,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.base_num_frames = base_num_frames
        self.base_height = base_height
        self.base_width = base_width
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.theta = theta

    def _prepare_video_coords(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        rope_interpolation_scale: Tuple[torch.Tensor, float, float],
        device: torch.device,
    ) -> torch.Tensor:
        # Always compute rope in fp32
        grid_h = torch.arange(height, dtype=torch.float32, device=device)
        grid_w = torch.arange(width, dtype=torch.float32, device=device)
        grid_f = torch.arange(num_frames, dtype=torch.float32, device=device)
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        if rope_interpolation_scale is not None:
            grid[:, 0:1] = grid[:, 0:1] * rope_interpolation_scale[0] * self.patch_size_t / self.base_num_frames
            grid[:, 1:2] = grid[:, 1:2] * rope_interpolation_scale[1] * self.patch_size / self.base_height
            grid[:, 2:3] = grid[:, 2:3] * rope_interpolation_scale[2] * self.patch_size / self.base_width

        grid = grid.flatten(2, 4).transpose(1, 2)

        return grid

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        rope_interpolation_scale: Optional[Tuple[torch.Tensor, float, float]] = None,
        video_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.size(0)

        if video_coords is None:
            grid = self._prepare_video_coords(
                batch_size,
                num_frames,
                height,
                width,
                rope_interpolation_scale=rope_interpolation_scale,
                device=hidden_states.device,
            )
        else:
            grid = torch.stack(
                [
                    video_coords[:, 0] / self.base_num_frames,
                    video_coords[:, 1] / self.base_height,
                    video_coords[:, 2] / self.base_width,
                ],
                dim=-1,
            )

        start = 1.0
        end = self.theta
        freqs = self.theta ** torch.linspace(
            math.log(start, self.theta),
            math.log(end, self.theta),
            self.dim // 6,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * (grid.unsqueeze(-1) * 2 - 1)
        freqs = freqs.transpose(-1, -2).flatten(2)

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        if self.dim % 6 != 0:
            cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % 6])
            sin_padding = torch.zeros_like(cos_freqs[:, :, : self.dim % 6])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs


@maybe_allow_in_graph
class LTXVideoTransformerBlock(nn.Module):
    r"""
    Transformer block used in [LTX](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization layer to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = LTXAttention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
        )

        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = LTXAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
        )

        self.ff = FeedForward(dim, activation_fn=activation_fn)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None].to(temb.device) + temb.reshape(
            batch_size, temb.size(1), num_ada_params, -1
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        attn_hidden_states = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=None,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + attn_hidden_states
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


@maybe_allow_in_graph
class LTXVideoTransformer3DModel(
    ModelMixin, ConfigMixin, AttentionMixin, FromOriginalModelMixin, PeftAdapterMixin, CacheMixin
):
    r"""
    A Transformer model for video-like data used in [LTX](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        in_channels (`int`, defaults to `128`):
            The number of channels in the input.
        out_channels (`int`, defaults to `128`):
            The number of channels in the output.
        patch_size (`int`, defaults to `1`):
            The size of the spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of the tmeporal patches to use in the patch embedding layer.
        num_attention_heads (`int`, defaults to `32`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        cross_attention_dim (`int`, defaults to `2048 `):
            The number of channels for cross attention heads.
        num_layers (`int`, defaults to `28`):
            The number of layers of Transformer blocks to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        qk_norm (`str`, defaults to `"rms_norm_across_heads"`):
            The normalization layer to use.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["norm"]
    _repeated_blocks = ["LTXVideoTransformerBlock"]
    _cp_plan = {
        "": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_attention_mask": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
        "rope": {
            0: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
            1: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 2048,
        num_layers: int = 28,
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 4096,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        action_expert: bool = False,
        action_scorer: bool = False,
        action_strategy_conditioning: bool = False,
        other_agent_expert: bool = False,
        other_agent_num_layers: int = 8,
        other_agent_horizon_steps: int = 8,
        other_agent_in_channels: int = 4,
        other_agent_out_channels: int = 4,
        other_agent_num_attention_heads: int = 16,
        other_agent_attention_head_dim: int = 32,
        other_agent_position_scale: float = 80.0,
        other_agent_velocity_scale: float = 30.0,
        other_agent_size_scale: float = 10.0,
        strategy_max_reasoning_level: int = 3,
        strategy_interval_length: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim

        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.time_embed = AdaLayerNormSingle(inner_dim, use_additional_conditions=False)

        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)

        self.rope = LTXVideoRotaryPosEmbed(
            dim=inner_dim,
            base_num_frames=20,
            base_height=2048,
            base_width=2048,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            theta=10000.0,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                LTXVideoTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels)

        self.gradient_checkpointing = False


        self.action_expert = action_expert
        if action_strategy_conditioning and not self.action_expert:
            raise ValueError(
                "action_strategy_conditioning=True requires action_expert=True."
            )
        if self.action_expert:
            add_action_expert(
                self,
                num_layers=num_layers,
                inner_dim=inner_dim,
                activation_fn=activation_fn,
                norm_eps=norm_eps,
                attention_bias=attention_bias,
                norm_elementwise_affine=norm_elementwise_affine,
                attention_out_bias=attention_out_bias,
                qk_norm=qk_norm,
                attention_class=Attention,
                attention_processor=LTXVideoAttentionProcessor2_0(),
                action_strategy_conditioning=action_strategy_conditioning,
                **kwargs
            )

        self.action_scorer = action_scorer
        if self.action_scorer:
            self.traj_scorer = ActionTrajectoryScorer()

        self.has_other_agent_expert = other_agent_expert
        other_hidden_dim = None
        if self.has_other_agent_expert:
            other_hidden_dim = (
                other_agent_num_attention_heads
                * other_agent_attention_head_dim
            )
            self.other_agent_model = SharedOtherAgentExpert(
                horizon_steps=other_agent_horizon_steps,
                input_dim=other_agent_in_channels,
                output_dim=other_agent_out_channels,
                hidden_dim=other_hidden_dim,
                video_hidden_dim=inner_dim,
                num_heads=other_agent_num_attention_heads,
                head_dim=other_agent_attention_head_dim,
                num_layers=other_agent_num_layers,
                num_video_layers=num_layers,
                state_position_scale=other_agent_position_scale,
                state_velocity_scale=other_agent_velocity_scale,
                state_size_scale=other_agent_size_scale,
            )

        self.has_strategy_encoder = (
            action_strategy_conditioning or self.has_other_agent_expert
        )
        if self.has_strategy_encoder:
            if action_strategy_conditioning:
                strategy_embedding_dim = self.action_inner_dim
            else:
                strategy_embedding_dim = other_hidden_dim
            if (
                action_strategy_conditioning
                and self.has_other_agent_expert
                and self.action_inner_dim != other_hidden_dim
            ):
                raise ValueError(
                    "Action and Other-Agent Experts must share one strategy "
                    "embedding dimension."
                )
            self.strategy_encoder = StrategyTokenEncoder(
                horizon_steps=other_agent_horizon_steps,
                embedding_dim=strategy_embedding_dim,
                max_reasoning_level=strategy_max_reasoning_level,
                position_scale=other_agent_position_scale,
                velocity_scale=other_agent_velocity_scale,
                size_scale=other_agent_size_scale,
            )
            self.level_zero_policy = LevelZeroPolicy(
                horizon_steps=other_agent_horizon_steps,
                interval_length=strategy_interval_length,
            )


    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def encode_level_strategy(
        self,
        *,
        strategies: StrategyTrajectoryBatch,
        ego_velocity: torch.Tensor,
        agent_states: AgentStateBatch,
        reasoning_level: int | torch.Tensor,
    ) -> StrategyTokenBatch:
        """Encode one completed strategy level for the next best response."""

        if not self.has_strategy_encoder:
            raise ValueError(
                "Strategy encoding requires a strategy-conditioned expert."
            )
        return self.strategy_encoder(
            strategies=strategies,
            ego_velocity=ego_velocity,
            agent_states=agent_states,
            reasoning_level=reasoning_level,
        )

    def encode_causal_level_zero_strategy(
        self,
        *,
        agent_states: AgentStateBatch,
        context_dict: Dict[str, torch.Tensor],
        require_references: bool = False,
    ) -> StrategyTokenBatch:
        """Encode Level-0 using current states only, never future labels."""

        if context_dict is None or context_dict.get("vel") is None:
            raise ValueError(
                "Causal Level-0 strategy construction requires "
                "context_dict['vel']."
            )
        ego_velocity = context_dict["vel"].to(
            device=agent_states.current_states.device,
            dtype=agent_states.current_states.dtype,
        )
        reference_keys = (
            "level_zero_ego_reference",
            "level_zero_ego_reference_mask",
            "level_zero_other_references",
            "level_zero_other_reference_mask",
        )
        present_reference_keys = [
            name for name in reference_keys if context_dict.get(name) is not None
        ]
        references = None
        if require_references and not present_reference_keys:
            raise ValueError(
                "map_lane_following requires the complete Level-0 map "
                "reference contract."
            )
        if present_reference_keys:
            if len(present_reference_keys) != len(reference_keys):
                missing = sorted(set(reference_keys) - set(present_reference_keys))
                raise ValueError(
                    "Level-0 map references must be provided together; missing: "
                    + ", ".join(missing)
                )
            references = LevelZeroReferenceBatch(
                ego_trajectory=context_dict[
                    "level_zero_ego_reference"
                ].to(device=agent_states.current_states.device, dtype=agent_states.current_states.dtype),
                ego_reference_mask=context_dict[
                    "level_zero_ego_reference_mask"
                ].to(device=agent_states.current_states.device, dtype=torch.bool),
                other_trajectories=context_dict[
                    "level_zero_other_references"
                ].to(device=agent_states.current_states.device, dtype=agent_states.current_states.dtype),
                other_reference_mask=context_dict[
                    "level_zero_other_reference_mask"
                ].to(device=agent_states.current_states.device, dtype=torch.bool),
            )
        level_zero_strategies = self.level_zero_policy(
            ego_velocity,
            agent_states,
            references,
        )
        return self.encode_level_strategy(
            strategies=level_zero_strategies,
            ego_velocity=ego_velocity,
            agent_states=agent_states,
            reasoning_level=0,
        )

    def encode_causal_predicted_strategy(
        self,
        *,
        ego_strategy: torch.Tensor,
        other_strategies: torch.Tensor,
        other_strategy_mask: torch.Tensor | None = None,
        source_level: int | torch.Tensor,
        agent_states: AgentStateBatch,
        context_dict: Dict[str, torch.Tensor],
    ) -> StrategyTokenBatch:
        """Encode a detached previous-level prediction for the next level."""

        if context_dict is None or context_dict.get("vel") is None:
            raise ValueError(
                "Predicted strategy construction requires "
                "context_dict['vel']."
            )
        strategies = StrategyTrajectoryBatch(
            ego_trajectory=ego_strategy,
            other_trajectories=other_strategies,
            agent_mask=agent_states.agent_mask,
            future_mask=other_strategy_mask,
        )
        strategies.validate(
            horizon_steps=self.other_agent_model.horizon_steps,
        )
        ego_velocity = context_dict["vel"].to(
            device=agent_states.current_states.device,
            dtype=agent_states.current_states.dtype,
        )
        return self.encode_level_strategy(
            strategies=strategies,
            ego_velocity=ego_velocity,
            agent_states=agent_states,
            reasoning_level=source_level,
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        rope_interpolation_scale: Optional[Union[Tuple[float, float, float], torch.Tensor]] = None,
        video_coords: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        action_states: torch.Tensor = None,
        action_timestep: torch.LongTensor = None,
        return_video: bool = True,
        return_action: bool = False,
        store_buffer=False,
        video_states_buffer=None,
        video_attention_mask: torch.Tensor = None,
        context_dict: Optional[Dict[str, torch.Tensor]] = None,
        other_agent_states: torch.Tensor = None,
        other_agent_timestep: torch.Tensor = None,
        other_agent_current_states: torch.Tensor = None,
        other_agent_types: torch.Tensor = None,
        other_agent_mask: torch.Tensor = None,
        strategy_tokens: torch.Tensor = None,
        strategy_token_mask: torch.Tensor = None,
        return_other_agents: bool = False,
        build_level_zero_strategy: bool = False,
        require_level_zero_references: bool = False,
        previous_ego_strategy: torch.Tensor = None,
        previous_other_strategies: torch.Tensor = None,
        previous_other_strategy_mask: torch.Tensor = None,
        strategy_source_level: int | torch.Tensor = None,
    ) -> torch.Tensor:  
        has_previous_strategy = (
            previous_ego_strategy is not None
            or previous_other_strategies is not None
            or previous_other_strategy_mask is not None
            or strategy_source_level is not None
        )
        if (build_level_zero_strategy or has_previous_strategy) and (
            not return_action
            or not return_other_agents
            or not getattr(self, "action_strategy_conditioning", False)
        ):
            raise ValueError(
                "Joint causal Level-k training requires return_action=True, "
                "return_other_agents=True, and "
                "action_strategy_conditioning=True."
            )
        
        if return_video or store_buffer:

            if store_buffer:
                video_states_buffer = []
            image_rotary_emb = self.rope(
                hidden_states, num_frames, height, width, rope_interpolation_scale, video_coords
                )

            # convert encoder_attention_mask to a bias the same way we do for attention_mask
            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
            if video_attention_mask is not None and video_attention_mask.ndim == 2:
                video_attention_mask = (1 - video_attention_mask.to(hidden_states.dtype)) * -10000.0
                video_attention_mask = video_attention_mask.unsqueeze(0) 

            batch_size = hidden_states.size(0)
            hidden_states = self.proj_in(hidden_states)

            temb, embedded_timestep = self.time_embed(
                timestep.flatten(),
                batch_size=batch_size,
                hidden_dtype=hidden_states.dtype,
            )

            temb = temb.view(batch_size, -1, temb.size(-1))
            embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))
        
        other_agent_batch = None
        if return_other_agents:
            if not self.has_other_agent_expert:
                raise ValueError(
                    "return_other_agents=True requires other_agent_expert=True."
                )
            if video_states_buffer is None and not (store_buffer or return_video):
                raise ValueError(
                    "Other-Agent Expert requires current Video states or "
                    "video_states_buffer."
                )
            required_other_inputs = {
                "other_agent_states": other_agent_states,
                "other_agent_timestep": other_agent_timestep,
                "other_agent_current_states": other_agent_current_states,
                "other_agent_mask": other_agent_mask,
            }
            missing_other_inputs = [
                name for name, value in required_other_inputs.items()
                if value is None
            ]
            if missing_other_inputs:
                raise ValueError(
                    "Missing Other-Agent Expert inputs: "
                    + ", ".join(missing_other_inputs)
                )
            if other_agent_types is None:
                other_agent_types = torch.zeros(
                    other_agent_mask.shape,
                    device=other_agent_mask.device,
                    dtype=torch.long,
                )
            other_agent_batch = AgentStateBatch(
                current_states=other_agent_current_states,
                agent_types=other_agent_types,
                agent_mask=other_agent_mask,
            )
            other_agent_batch.validate()

        strategy_token_batch = None
        requires_strategy_tokens = return_other_agents or (
            return_action
            and getattr(self, "action_strategy_conditioning", False)
        )
        if requires_strategy_tokens:
            explicit_strategy = (
                strategy_tokens is not None
                or strategy_token_mask is not None
            )
            strategy_modes = sum(
                (
                    bool(explicit_strategy),
                    bool(build_level_zero_strategy),
                    bool(has_previous_strategy),
                )
            )
            if strategy_modes > 1:
                raise ValueError(
                    "Use exactly one strategy source: explicit tokens, "
                    "Level-0, or a previous prediction."
                )
            if explicit_strategy:
                if strategy_tokens is None or strategy_token_mask is None:
                    raise ValueError(
                        "strategy_tokens and strategy_token_mask must be "
                        "provided together."
                    )
                strategy_token_batch = StrategyTokenBatch(
                    tokens=strategy_tokens,
                    token_mask=strategy_token_mask,
                )
            elif build_level_zero_strategy:
                if other_agent_batch is None:
                    raise ValueError(
                        "Causal Level-0 strategy construction requires "
                        "current Other-agent states."
                    )
                strategy_token_batch = (
                    self.encode_causal_level_zero_strategy(
                        agent_states=other_agent_batch,
                        context_dict=context_dict,
                        require_references=require_level_zero_references,
                    )
                )
            elif has_previous_strategy:
                if (
                    previous_ego_strategy is None
                    or previous_other_strategies is None
                    or strategy_source_level is None
                ):
                    raise ValueError(
                        "Previous predicted strategy requires Ego, Other, "
                        "and source-level values together."
                    )
                if other_agent_batch is None:
                    raise ValueError(
                        "Previous predicted strategy requires current "
                        "Other-agent states."
                    )
                strategy_token_batch = (
                    self.encode_causal_predicted_strategy(
                        ego_strategy=previous_ego_strategy,
                        other_strategies=previous_other_strategies,
                        other_strategy_mask=previous_other_strategy_mask,
                        source_level=strategy_source_level,
                        agent_states=other_agent_batch,
                        context_dict=context_dict,
                    )
                )
            else:
                raise ValueError(
                    "Strategy-conditioned experts require explicit strategy "
                    "tokens or build_level_zero_strategy=True."
                )
            strategy_token_batch.validate()

        ego_strategy_context = None
        if (
            return_action
            and getattr(self, "action_strategy_conditioning", False)
        ):
            ego_strategy_context = prepare_ego_strategy_context(
                strategy_token_batch
            )

        if return_action:
            ### when video_states_buffer is not None, action blocks will directly use the input buffers
            ### when video_states_buffer is None, store_buffer should be true to save video buffers
            if video_states_buffer is None:
                assert store_buffer or return_video
            action_temb_fused, action_embedded_timestep, action_rotary_emb, action_hidden_states = preprocessing_action_states(self, action_states, action_timestep,context_dict)
            if (
                ego_strategy_context is not None
                and ego_strategy_context.tokens.dtype != action_hidden_states.dtype
            ):
                # Under autocast, Action latents may be lower precision than
                # the strategy encoder output.  The conditioner intentionally
                # enforces a shared dtype before attention, so adapt only the
                # floating-point context while preserving its masks.
                ego_strategy_context = EgoStrategyContext(
                    tokens=ego_strategy_context.tokens.to(
                        dtype=action_hidden_states.dtype
                    ),
                    visibility_mask=ego_strategy_context.visibility_mask,
                    has_valid_other=ego_strategy_context.has_valid_other,
                )
                ego_strategy_context.validate()

        if return_other_agents:
            # The diffusion branch can be autocast to a lower precision while
            # strategy/state tensors are still in their source dtype (usually
            # float32).  The Other-Agent Expert intentionally requires all
            # floating-point inputs to share one dtype, so adapt the auxiliary
            # inputs at this model boundary.  Keep masks and categorical agent
            # types unchanged.
            other_agent_dtype = other_agent_states.dtype
            if (
                other_agent_batch.current_states.dtype != other_agent_dtype
                or strategy_token_batch.tokens.dtype != other_agent_dtype
            ):
                other_agent_batch = AgentStateBatch(
                    current_states=other_agent_batch.current_states.to(
                        dtype=other_agent_dtype
                    ),
                    agent_types=other_agent_batch.agent_types,
                    agent_mask=other_agent_batch.agent_mask,
                )
                strategy_token_batch = StrategyTokenBatch(
                    tokens=strategy_token_batch.tokens.to(
                        dtype=other_agent_dtype
                    ),
                    token_mask=strategy_token_batch.token_mask,
                )
                # Re-validate after adaptation so shape/device/mask contracts
                # remain enforced before entering the expert.
                other_agent_batch.validate()
                strategy_token_batch.validate()
            (
                other_agent_hidden_states,
                other_agent_conditioning,
                other_agent_strategy_visibility,
            ) = self.other_agent_model.preprocess(
                noisy_trajectories=other_agent_states,
                timestep=other_agent_timestep,
                agent_states=other_agent_batch,
                strategy_tokens=strategy_token_batch,
            )

        def _align_other_agent_block_inputs(
            video_hidden_states: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, StrategyTokenBatch]:
            """Match auxiliary block inputs to the current Video dtype."""

            target_dtype = video_hidden_states.dtype
            aligned_hidden_states = other_agent_hidden_states.to(
                dtype=target_dtype
            )
            aligned_conditioning = other_agent_conditioning.to(
                dtype=target_dtype
            )
            if strategy_token_batch.tokens.dtype == target_dtype:
                aligned_strategy_tokens = strategy_token_batch
            else:
                aligned_strategy_tokens = StrategyTokenBatch(
                    tokens=strategy_token_batch.tokens.to(dtype=target_dtype),
                    token_mask=strategy_token_batch.token_mask,
                )
            return (
                aligned_hidden_states,
                aligned_conditioning,
                aligned_strategy_tokens,
            )

        for block_idx, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                if return_video or store_buffer:
                    hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        encoder_attention_mask,
                    )
                    if store_buffer:
                        video_states_buffer.append(hidden_states.clone())
                else:
                    hidden_states = video_states_buffer[block_idx]

                if return_action:
                    ### final_hidden_states:  video features, b (v t h w) c
                    ### action_hidden_states: random actions, b v c
                    ### 
                    final_hidden_states = hidden_states
                    action_hidden_states = self.action_blocks[block_idx](
                        hidden_states=action_hidden_states,
                        encoder_hidden_states=final_hidden_states,
                        temb=action_temb_fused,
                        rotary_emb=action_rotary_emb,
                        strategy_context=ego_strategy_context,
                    )
                if (
                    return_other_agents
                    and block_idx
                    in self.other_agent_model.video_to_expert_index
                ):
                    other_block_index = (
                        self.other_agent_model.video_to_expert_index[block_idx]
                    )
                    (
                        other_agent_hidden_states,
                        other_agent_conditioning,
                        strategy_token_batch,
                    ) = _align_other_agent_block_inputs(hidden_states)
                    other_agent_hidden_states = (
                        self.other_agent_model.forward_block(
                            expert_block_index=other_block_index,
                            hidden_states=other_agent_hidden_states,
                            conditioning=other_agent_conditioning,
                            video_hidden_states=hidden_states,
                            strategy_tokens=strategy_token_batch,
                            strategy_visibility=other_agent_strategy_visibility,
                            agent_mask=other_agent_mask,
                        )
                    )
            else:
                if return_video or store_buffer:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                    if store_buffer:
                        video_states_buffer.append(hidden_states.clone())
                else:
                    hidden_states = video_states_buffer[block_idx]

                if return_action:
                    ### final_hidden_states:  video features, b (v t h w) c
                    ### action_hidden_states: random actions, b v c
                    ### 
                    final_hidden_states = hidden_states # rearrange(hidden_states, '(b v) l c -> b (v l) c', v=n_view)
                    
                    action_hidden_states = self.action_blocks[block_idx](
                        hidden_states=action_hidden_states,
                        encoder_hidden_states=final_hidden_states,
                        temb=action_temb_fused,
                        rotary_emb=action_rotary_emb,
                        strategy_context=ego_strategy_context,
                    )
                if (
                    return_other_agents
                    and block_idx
                    in self.other_agent_model.video_to_expert_index
                ):
                    other_block_index = (
                        self.other_agent_model.video_to_expert_index[block_idx]
                    )
                    (
                        other_agent_hidden_states,
                        other_agent_conditioning,
                        strategy_token_batch,
                    ) = _align_other_agent_block_inputs(hidden_states)
                    other_agent_hidden_states = (
                        self.other_agent_model.forward_block(
                            expert_block_index=other_block_index,
                            hidden_states=other_agent_hidden_states,
                            conditioning=other_agent_conditioning,
                            video_hidden_states=hidden_states,
                            strategy_tokens=strategy_token_batch,
                            strategy_visibility=other_agent_strategy_visibility,
                            agent_mask=other_agent_mask,
                        )
                    )
        
        final_output = {}

        if store_buffer:
            final_output['video_states_buffer'] = video_states_buffer
        
        if return_video:
            scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
            shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

            hidden_states = self.norm_out(hidden_states)
            hidden_states = hidden_states * (1 + scale) + shift
            output = self.proj_out(hidden_states)

            final_output['video'] = output

        
        if return_action:
            if self.action_final_embeddings:
                action_scale_shift_values = self.action_scale_shift_table[None, None] + action_embedded_timestep[:, :, None]
                action_shift, action_scale = action_scale_shift_values[:,:,0], action_scale_shift_values[:,:,1]
                action_hidden_states = self.action_norm_out(action_hidden_states)
                action_hidden_states = action_hidden_states * (1 + action_scale) + action_shift
            else:
                action_hidden_states = self.action_norm_out(action_hidden_states)
                action_hidden_states = self.action_proj_extra(action_hidden_states)


            action_output = self.action_proj_out(action_hidden_states)

            final_output['action'] = action_output

        if return_other_agents:
            final_output["other_agent"] = self.other_agent_model.postprocess(
                other_agent_hidden_states,
                other_agent_mask,
            )

        if not return_dict:
            return (final_output,)

        return Transformer2DModelOutput(sample=final_output)





def apply_rotary_emb(x, freqs):
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)  # [B, S, C // 2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out
