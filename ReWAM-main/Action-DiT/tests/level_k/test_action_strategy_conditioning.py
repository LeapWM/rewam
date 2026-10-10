"""Tests for Ego Action Expert strategy-token conditioning."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from torch import nn

from navsim.agents.videodrive.level_k import (
    CompatibleRMSNorm,
    GatedStrategyConditioner,
    StrategyTokenBatch,
    prepare_ego_strategy_context,
)


def _strategy_batch(
    *,
    batch_size: int = 2,
    num_agents: int = 3,
    hidden_dim: int = 32,
    dtype: torch.dtype = torch.float32,
) -> StrategyTokenBatch:
    generator = torch.Generator().manual_seed(17)
    mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
    token_mask = torch.cat(
        (torch.ones(batch_size, 1, dtype=torch.bool), mask),
        dim=1,
    )
    tokens = torch.randn(
        batch_size,
        num_agents + 1,
        hidden_dim,
        generator=generator,
        dtype=dtype,
    )
    return StrategyTokenBatch(tokens=tokens, token_mask=token_mask)


def _conditioner(
    *,
    hidden_dim: int = 32,
) -> GatedStrategyConditioner:
    return GatedStrategyConditioner(
        hidden_dim=hidden_dim,
        context_dim=hidden_dim,
        num_heads=4,
        head_dim=hidden_dim // 4,
    )


def _load_exact_action_block():
    source_path = (
        Path(__file__).parents[2]
        / "navsim"
        / "agents"
        / "videodrive"
        / "video_models"
        / "diffusion_planner"
        / "diffusion_planner.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ActionTransformerBlock"
    )
    class_node.decorator_list = []

    class StubRMSNorm(CompatibleRMSNorm):
        def __init__(
            self,
            hidden_dim,
            eps=1e-6,
            elementwise_affine=False,
        ):
            super().__init__(hidden_dim, eps=eps)

    class StubAttention(nn.Module):
        def __init__(self, query_dim, **_kwargs):
            super().__init__()
            self.projection = nn.Linear(query_dim, query_dim)

        def forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=None,
            attention_mask=None,
        ):
            return self.projection(hidden_states)

    class StubSwiGLUFFN(nn.Module):
        def __init__(self, hidden_dim, bias=True):
            super().__init__()
            self.projection = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        def forward(self, hidden_states):
            return self.projection(hidden_states)

    namespace = {
        "torch": torch,
        "nn": nn,
        "Optional": __import__("typing").Optional,
        "Tuple": __import__("typing").Tuple,
        "RMSNorm": StubRMSNorm,
        "SwiGLUFFN": StubSwiGLUFFN,
        "GatedStrategyConditioner": GatedStrategyConditioner,
        "EgoStrategyContext": __import__(
            "navsim.agents.videodrive.level_k",
            fromlist=["EgoStrategyContext"],
        ).EgoStrategyContext,
    }
    module = ast.Module(body=[class_node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["ActionTransformerBlock"], StubAttention


def test_prepare_context_removes_ego_and_builds_fallback() -> None:
    strategies = _strategy_batch()
    context = prepare_ego_strategy_context(strategies)
    assert context.tokens.shape == (2, 4, 32)
    torch.testing.assert_close(context.tokens[:, :3], strategies.other_tokens)
    assert torch.count_nonzero(context.tokens[:, -1]) == 0
    assert torch.all(context.has_valid_other)
    assert not torch.any(context.visibility_mask[:, -1])

    strategies.token_mask[1, 1:] = False
    strategies.tokens[1, 1:] = 0.0
    empty_context = prepare_ego_strategy_context(strategies)
    assert not empty_context.has_valid_other[1]
    assert empty_context.visibility_mask[1, -1]
    assert torch.count_nonzero(empty_context.tokens[1]) == 0


def test_changing_ego_token_does_not_change_context() -> None:
    strategies = _strategy_batch()
    original = prepare_ego_strategy_context(strategies)
    changed_tokens = strategies.tokens.clone()
    changed_tokens[:, 0] += 100.0
    changed = prepare_ego_strategy_context(
        StrategyTokenBatch(changed_tokens, strategies.token_mask)
    )
    torch.testing.assert_close(changed.tokens, original.tokens)
    torch.testing.assert_close(changed.visibility_mask, original.visibility_mask)


def test_zero_gate_is_exact_identity() -> None:
    conditioner = _conditioner().eval()
    hidden = torch.randn(2, 8, 32)
    context = prepare_ego_strategy_context(_strategy_batch())
    output = conditioner(hidden, context)
    assert torch.equal(output, hidden)
    assert torch.count_nonzero(conditioner.gate) == 0


def test_nonzero_gate_uses_valid_other_tokens() -> None:
    conditioner = _conditioner().eval()
    conditioner.gate.data.fill_(1.0)
    hidden = torch.randn(2, 8, 32)
    strategies = _strategy_batch()
    context = prepare_ego_strategy_context(strategies)
    original = conditioner(hidden, context)
    changed_tokens = strategies.tokens.clone()
    changed_tokens[:, 1] += 50.0
    changed_context = prepare_ego_strategy_context(
        StrategyTokenBatch(changed_tokens, strategies.token_mask)
    )
    changed = conditioner(hidden, changed_context)
    assert not torch.allclose(changed, original)


def test_empty_other_set_is_exact_identity_without_nan() -> None:
    conditioner = _conditioner().eval()
    conditioner.gate.data.fill_(1.0)
    hidden = torch.randn(2, 8, 32)
    strategies = _strategy_batch()
    strategies.token_mask[:, 1:] = False
    strategies.tokens[:, 1:] = 0.0
    context = prepare_ego_strategy_context(strategies)
    output = conditioner(hidden, context)
    assert torch.equal(output, hidden)
    assert torch.isfinite(output).all()


def test_padding_content_cannot_affect_output() -> None:
    conditioner = _conditioner().eval()
    conditioner.gate.data.fill_(1.0)
    hidden = torch.randn(2, 8, 32)
    strategies = _strategy_batch()
    strategies.token_mask[:, -1] = False
    strategies.tokens[:, -1] = 0.0
    original = conditioner(
        hidden,
        prepare_ego_strategy_context(strategies),
    )

    # StrategyTokenBatch intentionally requires padded tokens to remain zero.
    # Changing the pre-mask source then re-zeroing must preserve the result.
    changed_tokens = strategies.tokens.clone()
    changed_tokens[:, -1] = 1000.0
    changed_tokens[:, -1] = 0.0
    changed = conditioner(
        hidden,
        prepare_ego_strategy_context(
            StrategyTokenBatch(changed_tokens, strategies.token_mask)
        ),
    )
    torch.testing.assert_close(changed, original)


def test_other_token_permutation_does_not_change_ego_output() -> None:
    conditioner = _conditioner().eval()
    conditioner.gate.data.fill_(0.7)
    hidden = torch.randn(2, 8, 32)
    strategies = _strategy_batch()
    original = conditioner(
        hidden,
        prepare_ego_strategy_context(strategies),
    )
    permutation = torch.tensor([2, 0, 1])
    permuted = StrategyTokenBatch(
        tokens=torch.cat(
            (
                strategies.tokens[:, :1],
                strategies.other_tokens[:, permutation],
            ),
            dim=1,
        ),
        token_mask=torch.cat(
            (
                strategies.token_mask[:, :1],
                strategies.other_mask[:, permutation],
            ),
            dim=1,
        ),
    )
    permuted_output = conditioner(
        hidden,
        prepare_ego_strategy_context(permuted),
    )
    torch.testing.assert_close(permuted_output, original)


def test_gate_then_attention_gradient_flow() -> None:
    conditioner = _conditioner()
    hidden = torch.randn(2, 8, 32, requires_grad=True)
    context = prepare_ego_strategy_context(_strategy_batch())
    conditioner(hidden, context).square().mean().backward()
    assert conditioner.gate.grad is not None
    assert conditioner.gate.grad.abs().sum() > 0
    attention_grad = conditioner.attention.to_q.weight.grad
    assert attention_grad is not None
    assert torch.count_nonzero(attention_grad) == 0

    conditioner.zero_grad(set_to_none=True)
    conditioner.gate.data.fill_(0.1)
    conditioner(hidden.detach(), context).square().mean().backward()
    assert conditioner.attention.to_q.weight.grad is not None
    assert conditioner.attention.to_q.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_conditioner_dtype(dtype: torch.dtype) -> None:
    conditioner = _conditioner(hidden_dim=16).to(dtype=dtype)
    conditioner.gate.data.fill_(0.2)
    hidden = torch.randn(1, 4, 16, dtype=dtype)
    strategies = _strategy_batch(
        batch_size=1,
        num_agents=2,
        hidden_dim=16,
        dtype=dtype,
    )
    output = conditioner(
        hidden,
        prepare_ego_strategy_context(strategies),
    )
    assert output.dtype == dtype
    assert torch.isfinite(output).all()


def test_optional_conditioner_checkpoint_compatibility() -> None:
    class HostBlock(nn.Module):
        def __init__(self, enabled: bool) -> None:
            super().__init__()
            self.original_projection = nn.Linear(8, 8)
            if enabled:
                self.strategy_conditioner = GatedStrategyConditioner(
                    hidden_dim=8,
                    context_dim=8,
                    num_heads=2,
                    head_dim=4,
                )

    original = HostBlock(enabled=False)
    original_state = original.state_dict()
    disabled = HostBlock(enabled=False)
    disabled.load_state_dict(original_state, strict=True)

    enabled = HostBlock(enabled=True)
    result = enabled.load_state_dict(original_state, strict=False)
    assert result.unexpected_keys == []
    assert result.missing_keys
    assert all(
        key.startswith("strategy_conditioner.")
        for key in result.missing_keys
    )
    torch.testing.assert_close(
        enabled.original_projection.weight,
        original.original_projection.weight,
    )


def test_exact_action_block_zero_gate_preserves_old_output() -> None:
    action_block_class, attention_class = _load_exact_action_block()
    hidden_dim = 16
    common_kwargs = {
        "attention_class": attention_class,
        "attention_args": [
            {"query_dim": hidden_dim},
            {"query_dim": hidden_dim},
        ],
        "dim": hidden_dim,
        "num_attention_heads": 2,
        "attention_head_dim": 8,
    }
    original = action_block_class(
        **common_kwargs,
        strategy_conditioning=False,
    ).eval()
    conditioned = action_block_class(
        **common_kwargs,
        strategy_conditioning=True,
    ).eval()
    load_result = conditioned.load_state_dict(
        original.state_dict(),
        strict=False,
    )
    assert load_result.unexpected_keys == []
    assert load_result.missing_keys
    assert all(
        key.startswith("strategy_conditioner.")
        for key in load_result.missing_keys
    )

    hidden = torch.randn(2, 4, hidden_dim)
    video = torch.randn(2, 5, hidden_dim)
    temb = torch.randn(2, 4, hidden_dim * 6)
    context = prepare_ego_strategy_context(
        _strategy_batch(
            batch_size=2,
            num_agents=3,
            hidden_dim=hidden_dim,
        )
    )
    original_output = original(
        hidden,
        video,
        temb,
    )
    zero_gate_output = conditioned(
        hidden,
        video,
        temb,
        strategy_context=context,
    )
    assert torch.equal(zero_gate_output, original_output)

    conditioned.strategy_conditioner.gate.data.fill_(0.2)
    active_output = conditioned(
        hidden,
        video,
        temb,
        strategy_context=context,
    )
    assert not torch.allclose(active_output, original_output)
