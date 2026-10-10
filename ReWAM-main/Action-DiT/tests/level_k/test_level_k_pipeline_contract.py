import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import pytest
import torch

from navsim.agents.videodrive.level_k import (
    AgentStateBatch,
    EgoTrajectoryCodec,
    LevelKConfig,
    LevelKReasoner,
    LevelZeroReferenceBatch,
    OtherTrajectoryCodec,
    StrategyTokenBatch,
    synchronize_level_k_model_config,
)


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_PATH = (
    ROOT
    / "navsim"
    / "agents"
    / "videodrive"
    / "video_models"
    / "pipeline"
    / "pipeline_ltx_condition.py"
)
TRANSFORMER_PATH = (
    ROOT
    / "navsim"
    / "agents"
    / "videodrive"
    / "video_models"
    / "ltx_models"
    / "transformer_ltx.py"
)
AGENT_PATH = (
    ROOT / "navsim" / "agents" / "videodrive" / "videodrive_agent.py"
)


def _fake_randn_tensor(shape, *, device, dtype, generator=None):
    return torch.randn(
        shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )


def _fake_retrieve_timesteps(scheduler, num_steps, device, timesteps):
    del num_steps, device
    scheduler.reset()
    values = torch.as_tensor(timesteps, dtype=torch.float32)
    return values, len(values)


def _load_exact_pipeline_methods():
    tree = ast.parse(PIPELINE_PATH.read_text(encoding="utf-8"))
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LTXConditionPipeline"
    )
    selected_names = {
        "_coerce_level_k_config",
        "_repeat_level_k_batch",
        "_prepare_level_k_inputs",
        "_validate_level_k_request",
        "_run_level_k_reasoning",
    }
    selected = [
        node
        for node in original.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in selected_names
    ]
    extracted = ast.ClassDef(
        name="ExactPipelineLevelKMethods",
        bases=[],
        keywords=[],
        body=selected,
        decorator_list=[],
    )
    module = ast.Module(body=[extracted], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Any": Any,
        "Callable": Callable,
        "Dict": Dict,
        "List": List,
        "Mapping": Mapping,
        "Optional": Optional,
        "Union": Union,
        "torch": torch,
        "AgentStateBatch": AgentStateBatch,
        "EgoTrajectoryCodec": EgoTrajectoryCodec,
        "LevelKConfig": LevelKConfig,
        "LevelKReasoner": LevelKReasoner,
        "LevelZeroReferenceBatch": LevelZeroReferenceBatch,
        "OtherTrajectoryCodec": OtherTrajectoryCodec,
        "StrategyTokenBatch": StrategyTokenBatch,
        "randn_tensor": _fake_randn_tensor,
        "retrieve_timesteps": _fake_retrieve_timesteps,
    }
    exec(compile(module, str(PIPELINE_PATH), "exec"), namespace)
    return namespace["ExactPipelineLevelKMethods"]


class _FakeScheduler:
    def __init__(self):
        self.reset_count = 0
        self.step_index = 0

    def reset(self):
        self.reset_count += 1
        self.step_index = 0

    def step(self, model_output, timestep, sample, return_dict=False):
        del timestep, return_dict
        self.step_index += 1
        return (sample + 0.1 * model_output,)


class _FakeTransformer:
    def __init__(self):
        self.encoded_levels = []
        self.forward_sources = []
        self.action_expert = True
        self.action_strategy_conditioning = True
        self.has_other_agent_expert = True
        self.has_strategy_encoder = True
        self.other_agent_model = SimpleNamespace(horizon_steps=2)
        self.strategy_encoder = SimpleNamespace(max_reasoning_level=3)

    def cache_context(self, name):
        del name
        return nullcontext()

    def encode_level_strategy(
        self,
        *,
        strategies,
        ego_velocity,
        agent_states,
        reasoning_level,
    ):
        del strategies, ego_velocity
        self.encoded_levels.append(reasoning_level)
        batch_size, num_agents = agent_states.agent_mask.shape
        tokens = torch.full(
            (batch_size, num_agents + 1, 4),
            float(reasoning_level + 1),
        )
        mask = torch.cat(
            (
                torch.ones(batch_size, 1, dtype=torch.bool),
                agent_states.agent_mask,
            ),
            dim=1,
        )
        tokens = tokens * mask[..., None]
        return StrategyTokenBatch(tokens=tokens, token_mask=mask)

    def __call__(self, **kwargs):
        source = kwargs["strategy_tokens"][:, 0, 0]
        self.forward_sources.append(source.detach().clone())
        action = torch.ones_like(kwargs["action_states"]) * source[
            :, None, None
        ]
        other = torch.ones_like(kwargs["other_agent_states"]) * source[
            :, None, None, None
        ]
        return ({"action": action, "other_agent": other},)


ExactPipelineMethods = _load_exact_pipeline_methods()


class _PipelineHarness(ExactPipelineMethods):
    def __init__(self, *, guidance_scale=1.0):
        self.transformer = _FakeTransformer()
        self.scheduler_action = _FakeScheduler()
        self.scheduler_other_agent = _FakeScheduler()
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = guidance_scale > 1.0
        self.video_buffer_calls = 0

    def _build_level_k_video_buffer(self, **kwargs):
        self.video_buffer_calls += 1
        latents = kwargs["latents"]
        if self.do_classifier_free_guidance:
            latents = torch.cat((latents, latents), dim=0)
        return [latents], latents, torch.zeros(latents.shape[:2])


def _agents(*, empty_second=False):
    states = torch.zeros(2, 3, 8)
    states[..., 3] = 1.0
    states[..., 4] = 2.0
    states[..., 6:8] = torch.tensor([4.5, 2.0])
    mask = torch.ones(2, 3, dtype=torch.bool)
    if empty_second:
        mask[1] = False
        states[1] = 0
    types = torch.where(
        mask,
        torch.zeros_like(mask, dtype=torch.long),
        torch.full_like(mask, -1, dtype=torch.long),
    )
    return AgentStateBatch(states, types, mask)


def _run(harness, *, reasoning_level):
    config = LevelKConfig(
        enabled=True,
        reasoning_level=reasoning_level,
        max_reasoning_level=3,
        max_agents=3,
        horizon_steps=2,
        interval_length=0.5,
    )
    return harness._run_level_k_reasoning(
        config=config,
        ego_velocity=torch.tensor([[3.0, 0.0], [2.0, 0.0]]),
        agents=_agents(empty_second=True),
        base_action_noise=torch.zeros(2, 2, 3),
        latents=torch.zeros(2, 5, 4),
        conditioning_mask=torch.zeros(2, 5),
        prompt_embeds=torch.zeros(2, 3, 4),
        prompt_attention_mask=torch.ones(2, 3),
        timesteps=torch.tensor([2.0, 1.0]),
        action_timestep_spec=[2.0, 1.0],
        num_inference_steps=2,
        main_num_latent_frames=2,
        latent_height=1,
        latent_width=1,
        video_coords=torch.zeros(2, 3, 5),
        attention_kwargs=None,
        context_dict={"vel": torch.zeros(2, 2)},
        generator=torch.Generator().manual_seed(7),
        noise_seed=11,
    )


def test_exact_pipeline_recurses_and_resets_independent_schedulers():
    harness = _PipelineHarness()
    actions, output = _run(harness, reasoning_level=2)
    assert actions.shape == (2, 2, 3)
    assert len(output.ego_levels) == 3
    assert len(output.other_levels) == 3
    assert harness.transformer.encoded_levels == [0, 1]
    assert len(harness.transformer.forward_sources) == 4
    assert harness.scheduler_action.reset_count == 2
    assert harness.scheduler_other_agent.reset_count == 2
    assert harness.video_buffer_calls == 1
    assert torch.count_nonzero(output.other_levels[-1][1]) == 0


def test_exact_pipeline_k_zero_never_calls_transformer_or_schedulers():
    harness = _PipelineHarness()
    actions, output = _run(harness, reasoning_level=0)
    assert output.reasoning_level == 0
    assert harness.video_buffer_calls == 0
    assert harness.transformer.encoded_levels == []
    assert harness.transformer.forward_sources == []
    assert harness.scheduler_action.reset_count == 0
    assert harness.scheduler_other_agent.reset_count == 0
    physical = EgoTrajectoryCodec().decode(actions)
    torch.testing.assert_close(
        physical[0, :, 0],
        torch.tensor([1.5, 3.0]),
    )


def test_cfg_duplicates_strategy_conditions_without_changing_batch_output():
    harness = _PipelineHarness(guidance_scale=2.0)
    actions, output = _run(harness, reasoning_level=1)
    assert actions.shape == (2, 2, 3)
    assert output.ego_levels[-1].shape == (2, 2, 4)
    assert all(source.shape == (4,) for source in harness.transformer.forward_sources)


def test_pipeline_input_repetition_and_validation_use_exact_methods():
    harness = _PipelineHarness()
    config = LevelKConfig(
        enabled=True,
        reasoning_level=1,
        max_agents=3,
        horizon_steps=2,
    )
    ego, agents, references = harness._prepare_level_k_inputs(
        config=config,
        ego_velocity=torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
        other_agent_current_states=_agents().current_states,
        other_agent_types=_agents().agent_types,
        other_agent_mask=_agents().agent_mask,
        batch_size=2,
        num_videos_per_prompt=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert ego.shape == (4, 2)
    assert agents.current_states.shape == (4, 3, 8)
    assert references is None
    torch.testing.assert_close(ego[0], ego[1])
    torch.testing.assert_close(ego[2], ego[3])
    harness._validate_level_k_request(
        config=config,
        return_action=True,
        return_video=False,
        action_chunk=2,
        action_dim=3,
        callback_on_step_end=None,
    )


def test_pipeline_repeats_and_preserves_map_level_zero_references():
    harness = _PipelineHarness()
    config = LevelKConfig(
        enabled=True,
        reasoning_level=1,
        max_agents=3,
        horizon_steps=2,
        level_zero_mode="map_lane_following",
    )
    ego_reference = torch.zeros(2, 2, 4, dtype=torch.float64)
    ego_reference[..., 0] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ego_reference[..., 3] = 1.0
    other_reference = torch.zeros(2, 3, 2, 4, dtype=torch.float64)
    other_reference[..., 0] = 0.5
    other_reference[..., 3] = 1.0
    other_reference[1] = 0.0
    ego_mask = torch.tensor([True, False])
    other_mask = torch.tensor(
        [[True, False, True], [False, False, False]]
    )
    other_reference[~other_mask] = 0.0

    ego, agents, references = harness._prepare_level_k_inputs(
        config=config,
        ego_velocity=torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
        other_agent_current_states=_agents(empty_second=True).current_states,
        other_agent_types=_agents(empty_second=True).agent_types,
        other_agent_mask=_agents(empty_second=True).agent_mask,
        level_zero_ego_reference=ego_reference,
        level_zero_ego_reference_mask=ego_mask,
        level_zero_other_references=other_reference,
        level_zero_other_reference_mask=other_mask,
        batch_size=2,
        num_videos_per_prompt=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert ego.shape == (4, 2)
    assert references is not None
    assert references.ego_trajectory.dtype == torch.float32
    assert references.ego_reference_mask.dtype == torch.bool
    assert references.ego_reference_mask.tolist() == [True, True, False, False]
    assert references.other_reference_mask.tolist() == [
        [True, False, True],
        [True, False, True],
        [False, False, False],
        [False, False, False],
    ]
    torch.testing.assert_close(
        references.ego_trajectory[0],
        references.ego_trajectory[1],
    )
    torch.testing.assert_close(
        references.ego_trajectory[2],
        references.ego_trajectory[3],
    )
    references.validate(
        agent_mask=agents.agent_mask,
        horizon_steps=config.horizon_steps,
    )


def test_map_pipeline_rejects_completely_missing_level_zero_references():
    harness = _PipelineHarness()
    config = LevelKConfig(
        enabled=True,
        reasoning_level=1,
        max_agents=3,
        horizon_steps=2,
        level_zero_mode="map_lane_following",
    )
    with pytest.raises(ValueError, match="requires the complete"):
        harness._prepare_level_k_inputs(
            config=config,
            ego_velocity=torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
            other_agent_current_states=_agents().current_states,
            other_agent_types=_agents().agent_types,
            other_agent_mask=_agents().agent_mask,
            batch_size=2,
            num_videos_per_prompt=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_pipeline_source_contains_required_public_contracts():
    tree = ast.parse(PIPELINE_PATH.read_text(encoding="utf-8"))
    pipeline = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LTXConditionPipeline"
    )
    init = next(
        node
        for node in pipeline.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    infer = next(
        node
        for node in pipeline.body
        if isinstance(node, ast.FunctionDef) and node.name == "infer"
    )
    init_args = [arg.arg for arg in init.args.args]
    infer_args = [arg.arg for arg in infer.args.args]
    assert "scheduler_other_agent" in init_args
    assert {
        "level_k_config",
        "ego_velocity",
        "other_agent_current_states",
        "other_agent_types",
        "other_agent_mask",
    }.issubset(infer_args)


def test_transformer_registers_strategy_encoder_and_public_encode_method():
    tree = ast.parse(TRANSFORMER_PATH.read_text(encoding="utf-8"))
    transformer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LTXVideoTransformer3DModel"
    )
    init = next(
        node
        for node in transformer.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    method_names = {
        node.name
        for node in transformer.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "strategy_max_reasoning_level" in {
        argument.arg for argument in init.args.args
    }
    assert "encode_level_strategy" in method_names
    constructor_calls = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "StrategyTokenEncoder"
    ]
    assert len(constructor_calls) == 1


def test_agent_passes_current_only_level_k_inputs_to_pipeline():
    source = AGENT_PATH.read_text(encoding="utf-8")
    assert "compute_current_inputs(scene)" in source
    assert '"level_k_config": self.level_k_config' in source
    assert '"other_agent_current_states"' in source
    assert "Do not read future" in source


def test_yaml_defaults_preserve_original_inference_path():
    import yaml

    for name in (
        "video_model_infer_navsim.yaml",
        "video_model_infer_navsim_eval.yaml",
    ):
        path = (
            ROOT
            / "navsim"
            / "agents"
            / "videodrive"
            / "configs"
            / "ltx_model"
            / name
        )
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        model = config["diffusion_model"]["config"]
        level_k = config["level_k"]
        assert model["action_strategy_conditioning"] is False
        assert model["other_agent_expert"] is False
        assert model["strategy_max_reasoning_level"] == 3
        assert level_k["enabled"] is False
        assert level_k["max_reasoning_level"] == 3
        assert level_k["training_reasoning_level"] == 1


def test_yaml_level_k_switch_builds_a_complete_enabled_model_config():
    import copy
    import yaml

    path = (
        ROOT
        / "navsim"
        / "agents"
        / "videodrive"
        / "configs"
        / "ltx_model"
        / "video_model_infer_navsim.yaml"
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    model_config = copy.deepcopy(raw["diffusion_model"]["config"])
    level_k_values = dict(raw["level_k"])
    level_k_values["enabled"] = True
    config = LevelKConfig.from_dict(level_k_values)
    synchronize_level_k_model_config(config, model_config)

    assert model_config["action_strategy_conditioning"] is True
    assert model_config["other_agent_expert"] is True
    assert model_config["other_agent_horizon_steps"] == config.horizon_steps
    assert (
        model_config["strategy_max_reasoning_level"]
        == config.max_reasoning_level
    )


def test_other14_yaml_configs_preserve_28_layer_backbone() -> None:
    import yaml

    config_root = (
        ROOT
        / "navsim"
        / "agents"
        / "videodrive"
        / "configs"
        / "ltx_model"
    )
    for name in (
        "video_model_infer_navsim_other14.yaml",
        "video_model_infer_navsim_other14_eval.yaml",
    ):
        config = yaml.safe_load(
            (config_root / name).read_text(encoding="utf-8")
        )
        model = config["diffusion_model"]["config"]
        assert model["num_layers"] == 28
        assert model["other_agent_num_layers"] == 14
        assert model["other_agent_num_layers"] <= model["num_layers"]


def test_other28_yaml_configs_match_every_backbone_layer() -> None:
    import yaml

    config_root = (
        ROOT
        / "navsim"
        / "agents"
        / "videodrive"
        / "configs"
        / "ltx_model"
    )
    for name in (
        "video_model_infer_navsim_other28.yaml",
        "video_model_infer_navsim_other28_eval.yaml",
    ):
        config = yaml.safe_load(
            (config_root / name).read_text(encoding="utf-8")
        )
        model = config["diffusion_model"]["config"]
        assert model["num_layers"] == 28
        assert model["other_agent_num_layers"] == 28
        assert model["other_agent_num_layers"] == model["num_layers"]


def test_other28_training_yaml_enables_circle_collision_loss() -> None:
    import yaml

    config_path = (
        ROOT
        / "navsim"
        / "agents"
        / "videodrive"
        / "configs"
        / "ltx_model"
        / "video_model_infer_navsim_other28.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    level_k = config["level_k"]
    assert level_k["collision_loss_weight"] == 0.1
    assert level_k["collision_safety_margin"] == 0.3
    assert level_k["collision_temperature"] == 0.2
    assert level_k["collision_radius_scale"] == 0.7
    assert level_k["collision_loss_start_step"] == 1000
    assert level_k["collision_loss_ramp_steps"] == 4000
    assert level_k["ego_vehicle_length"] > 0.0
    assert level_k["ego_vehicle_width"] > 0.0
    assert level_k["ego_rear_axle_to_center"] >= 0.0
    assert level_k["progress_loss_weight"] == 0.1
    assert level_k["progress_tolerance_ratio"] == 0.05
    assert level_k["progress_min_gt_distance"] == 0.5
    assert level_k["progress_loss_start_step"] == 1000
    assert level_k["progress_loss_ramp_steps"] == 4000
    assert level_k["level_zero_mode"] == "map_lane_following"
    assert level_k["level_zero_fallback"] == "constant_heading"
    assert level_k["level_zero_lane_search_radius"] == 8.0
    assert level_k["level_zero_max_lateral_error"] == 4.0
    assert level_k["level_zero_route_search_depth"] == 30

    eval_config_path = config_path.with_name(
        "video_model_infer_navsim_other28_eval.yaml"
    )
    eval_level_k = yaml.safe_load(
        eval_config_path.read_text(encoding="utf-8")
    )["level_k"]
    assert eval_level_k["collision_loss_weight"] == 0.1
    assert eval_level_k["progress_loss_weight"] == 0.1
    assert eval_level_k["level_zero_mode"] == "map_lane_following"
    assert eval_level_k["level_zero_fallback"] == "constant_heading"


def test_agent_applies_level_k_model_config_before_model_initialization():
    source = AGENT_PATH.read_text(encoding="utf-8")
    sampling_validation = source.index(
        "validate_level_k_trajectory_sampling("
    )
    sync_call = source.index("synchronize_level_k_model_config(")
    model_load = source.index("load_diffusion_model(")
    assert sampling_validation < model_load
    assert sync_call < model_load
