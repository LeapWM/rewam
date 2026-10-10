#    Copyright (C) 2026 Xiaomi Corporation.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from typing import Any, Dict, List
import os
import torch
import argparse
import numpy as np
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.feature_target_builder import AbstractFeatureBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .surround_agent_features import SurroundAgentFeatureBuilder
from .videodrive_features import VideoDriveFeatureBuilder
from .level_k import (
    LevelKConfig,
    synchronize_level_k_model_config,
    validate_level_k_trajectory_sampling,
)
from yaml import load, Loader

from PIL import Image

from .utils.model_utils import (
    count_model_parameters,
    load_condition_models,
    load_diffusion_model,
)
from .utils.model_loader import load_vae
# ----------------------------------------------------
from .utils import import_custom_class
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video

import time


def _expand_config_values(value: Any) -> Any:
    """Expand ``$VAR``/``${VAR}`` and ``~`` in YAML string values."""

    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {key: _expand_config_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config_values(item) for item in value]
    return value


def _parse_env_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean environment value {value!r}; "
        "expected one of true/false, 1/0, yes/no."
    )


def _apply_level_k_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply inference-only Level-K overrides without editing the YAML."""

    level_k = dict(config.get("level_k") or {})
    bool_overrides = {
        "DRIVELAW_LEVEL_K_ENABLED": "enabled",
        "DRIVELAW_ORACLE_AGENT_STATES": "oracle_agent_states",
    }
    int_overrides = {
        "DRIVELAW_REASONING_LEVEL": "reasoning_level",
        "DRIVELAW_MAX_REASONING_LEVEL": "max_reasoning_level",
        "DRIVELAW_MAX_AGENTS": "max_agents",
        "DRIVELAW_HORIZON_STEPS": "horizon_steps",
        "DRIVELAW_LEVEL_ZERO_ROUTE_SEARCH_DEPTH": (
            "level_zero_route_search_depth"
        ),
    }
    float_overrides = {
        "DRIVELAW_AGENT_RANGE": "agent_range",
        "DRIVELAW_INTERVAL_LENGTH": "interval_length",
        "DRIVELAW_LEVEL_ZERO_LANE_SEARCH_RADIUS": (
            "level_zero_lane_search_radius"
        ),
        "DRIVELAW_LEVEL_ZERO_MAX_LATERAL_ERROR": (
            "level_zero_max_lateral_error"
        ),
        "DRIVELAW_LEVEL_ZERO_MAX_HEADING_ERROR": (
            "level_zero_max_heading_error"
        ),
    }

    for env_name, config_name in bool_overrides.items():
        if env_name in os.environ:
            level_k[config_name] = _parse_env_bool(os.environ[env_name])
    for env_name, config_name in int_overrides.items():
        if env_name in os.environ:
            level_k[config_name] = int(os.environ[env_name])
    for env_name, config_name in float_overrides.items():
        if env_name in os.environ:
            level_k[config_name] = float(os.environ[env_name])

    config["level_k"] = level_k
    return config

def _reject_unresolved_placeholders(config: Dict[str, Any]) -> None:
    """Fail early on the repository's old machine-specific placeholders."""

    checks = {
        "pretrained_model_name_or_path": config.get(
            "pretrained_model_name_or_path"
        ),
        "diffusion_model.model_path": (
            config.get("diffusion_model") or {}
        ).get("model_path"),
        "diffusion_model_class_path": config.get("diffusion_model_class_path"),
        "pipeline_class_path": config.get("pipeline_class_path"),
    }
    unresolved = {
        name: value
        for name, value in checks.items()
        if isinstance(value, str)
        and (
            "${" in value
            or "/path/to/" in value
            or value.startswith("PATH_TO_")
        )
    }
    if unresolved:
        details = "\n".join(f"  {name}: {value}" for name, value in unresolved.items())
        raise ValueError(
            "Unresolved DriveLaW configuration placeholders remain:\n"
            f"{details}\n"
            "Set the corresponding environment variables or edit the model YAML."
        )


class VideoDriveAgent(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        config_file,
        weight_dtype,
        device,
        view_mode: str = None,  # None: auto-detect; "front" for single view, "surround6" for multi-view
    ):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        with open(config_file, "r", encoding="utf-8") as stream:
            cd = load(stream, Loader=Loader) or {}
        cd = _expand_config_values(cd)
        cd = _apply_level_k_env_overrides(cd)
        _reject_unresolved_placeholders(cd)
        args = argparse.Namespace(**cd)
        self.args = args
        self.level_k_config = LevelKConfig.from_dict(
            getattr(args, "level_k", None)
        )
        validate_level_k_trajectory_sampling(
            self.level_k_config,
            num_poses=self._trajectory_sampling.num_poses,
            interval_length=self._trajectory_sampling.interval_length,
        )
        synchronize_level_k_model_config(
            self.level_k_config,
            args.diffusion_model["config"],
        )
        # Explicit current-agent annotations live on Scene rather than
        # AgentInput. Standard NAVSIM evaluation uses this flag to decide
        # whether it is allowed and necessary to pass Scene to the agent.
        self.requires_scene = bool(
            self.level_k_config.enabled
            and (
                self.level_k_config.oracle_agent_states
                or self.level_k_config.level_zero_mode
                == "map_lane_following"
            )
        )
        self._surround_agent_builder = None
        if self.level_k_config.enabled:
            self._surround_agent_builder = (
                SurroundAgentFeatureBuilder(
                    max_agents=self.level_k_config.max_agents,
                    agent_range=self.level_k_config.agent_range,
                    horizon_steps=self.level_k_config.horizon_steps,
                    interval_length=self.level_k_config.interval_length,
                    level_zero_mode=self.level_k_config.level_zero_mode,
                    level_zero_lane_search_radius=(
                        self.level_k_config.level_zero_lane_search_radius
                    ),
                    level_zero_max_lateral_error=(
                        self.level_k_config.level_zero_max_lateral_error
                    ),
                    level_zero_max_heading_error=(
                        self.level_k_config.level_zero_max_heading_error
                    ),
                    level_zero_route_search_depth=(
                        self.level_k_config.level_zero_route_search_depth
                    ),
                )
            )
        
        # Auto-detect view_mode (priority: explicit arg > config_file > path-based inference).
        if view_mode is None:
            if hasattr(args, 'view_mode') and args.view_mode:
                view_mode = args.view_mode
            elif 'multi_view' in str(config_file).lower() or 'multiview' in str(config_file).lower():
                view_mode = "surround6"
            else:
                view_mode = "front"
        
        # Tokenizers
        self.tokenizer = None

        # Text encoders
        self.text_encoder = None

        # Denoisers
        self.diffusion_model = None
        self.unet = None

        # Autoencoders
        self.vae = None

        # Scheduler
        self.scheduler = None

        self.pipe = None

        self.weight_dtype = torch.bfloat16
        self.device = "cuda" # device
        

        self.last_inference_timing = {}
        
        self.view_mode = view_mode
        print(f"[VideoDriveAgent] Using view_mode: {view_mode} (detected from config_file: {config_file})")


    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        print("Initializing models")
        device = self.device
        dtype = self.weight_dtype

        ### Load Tokenizer
        tokenizer_class = import_custom_class(
            self.args.tokenizer_class, getattr(self.args, "tokenizer_class_path", "transformers")
        )
        textenc_class = import_custom_class(
            self.args.textenc_class, getattr(self.args, "textenc_class_path", "transformers")
        )
        cond_models = load_condition_models(
            tokenizer_class, textenc_class,
            self.args.pretrained_model_name_or_path if not hasattr(self.args, "tokenizer_pretrained_model_name_or_path") else self.args.tokenizer_pretrained_model_name_or_path,
            load_weights=self.args.load_weights
        )
        self.tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
        self.text_encoder = text_encoder.to(device, dtype=dtype).eval()

        ### Load VAE
        vae_class = import_custom_class(
            self.args.vae_class, getattr(self.args, "vae_class_path", "transformers")
        )
        # if getattr(self.args, 'vae_path', False):
        #     self.vae = load_vae_models(vae_class, self.args.vae_path).to(device, dtype=dtype).eval()
        # else:
        #     self.vae = load_latent_models(vae_class, self.args.pretrained_model_name_or_path)["vae"].to(device, dtype=dtype).eval()
        self.vae = load_vae(self.args.pretrained_model_name_or_path).eval()

        if isinstance(self.vae.latents_mean, List):
            self.vae.latents_mean = torch.FloatTensor(self.vae.latents_mean)
        if isinstance(self.vae.latents_std, List):
            self.vae.latents_std = torch.FloatTensor(self.vae.latents_std)
        if self.vae is not None:
            if self.args.enable_slicing:
                self.vae.enable_slicing()
            if self.args.enable_tiling:
                self.vae.enable_tiling()
        self.SPATIAL_DOWN_RATIO = self.vae.spatial_compression_ratio
        self.TEMPORAL_DOWN_RATIO = self.vae.temporal_compression_ratio
        print(f'SPATIAL_DOWN_RATIO of VAE :{self.SPATIAL_DOWN_RATIO}')
        print(f'TEMPORAL_DOWN_RATIO of VAE :{self.TEMPORAL_DOWN_RATIO}')


        ### Load Diffusion Model
        diffusion_model_class = import_custom_class(
            self.args.diffusion_model_class, getattr(self.args, "diffusion_model_class_path", "transformers")
        )
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=self.args.diffusion_model['model_path'],
            load_weights=self.args.load_weights and getattr(self.args, "load_diffusion_model_weights", True),
            **self.args.diffusion_model['config']
        ).to(device, dtype=dtype)
        total_params = count_model_parameters(self.diffusion_model)
        print(f'Total parameters for transformer model:{total_params}')


        ### Load Diffuser Scheduler
        diffusion_scheduler_class = import_custom_class(
            self.args.diffusion_scheduler_class, getattr(self.args, "diffusion_scheduler_class_path", "diffusers")
        )
        if hasattr(self.args, "diffusion_scheduler_args"):
            self.scheduler = diffusion_scheduler_class(**self.args.diffusion_scheduler_args)
        else:
            self.scheduler = diffusion_scheduler_class()

        ### Import Inference Pipeline Class
        self.pipeline_class = import_custom_class(
            self.args.pipeline_class, getattr(self.args, "pipeline_class_path", "diffusers")
        )

        self.pipe = self.pipeline_class(
            self.scheduler, self.vae, self.text_encoder, self.tokenizer, self.diffusion_model
        )
        self.pipe.to(device)
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()

        self.vae_compression = self.pipe.vae_spatial_compression_ratio

    def round_to_vae_resolution(self, height, width):
        """Adjust resolution to be compatible with the VAE compression ratio."""
        height = height - (height % self.vae_compression)
        width = width - (width % self.vae_compression)
        return height, width
        
    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3,4,5,6,7,8,9,10,11])

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        if self.view_mode == "front":
            return [VideoDriveFeatureBuilder(
                view_mode="front",
            )]
        else:
            return [VideoDriveFeatureBuilder(
                view_mode="surround6",
                surround_keys=(
                    "cam_b0",  # back
                    "cam_l2",  # back-left
                    "cam_l0",  # front-left
                    "cam_f0",  # front
                    "cam_r0",  # front-right
                    "cam_r2",  # back-right
                ),
            )]

    def forward(
        self, features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Run the inference path; this release intentionally has no trainer."""
        return self.forward_test(features)

    def _pad_to_8n1(self, x_5d: torch.Tensor) -> torch.Tensor:
        T = x_5d.shape[2]
        need = (1 - (T % 8)) % 8  # 使得 (T + need) % 8 == 1
        if need > 0:
            tail = x_5d[:, :, -1:, :, :].repeat(1, 1, need, 1, 1)
            x_5d = torch.cat([x_5d, tail], dim=2)
        return x_5d

    @torch.inference_mode()
    def forward_test(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = self.device
        args = self.args
        
        timing = {}

        if "images" not in features:
            raise ValueError("forward_test expects features['images'] = (T,C,H,W) conditioning frames.")
        cond_tensor = features["images"]  # (T,3,H,W)
        
        if cond_tensor.ndim == 5:
            cond_tensor = cond_tensor.squeeze(0)
        elif cond_tensor.ndim == 3:
            cond_tensor = cond_tensor.unsqueeze(0)
        elif cond_tensor.ndim != 4:
            raise ValueError(f"Expected cond_tensor (T,C,H,W), got shape {cond_tensor.shape}")
        
        T_cond, _, H_in, W_in = cond_tensor.shape
        assert T_cond == 4
        weight_dtype = torch.bfloat16

        # hist = features["history_trajectory"].to(torch.float32).squeeze(0)
        # x0, y0, th0 = hist[0]         
        # xT, yT, thT = hist[-1]        

        # def _delta(a_last, a_first):
        #     return float((a_last - a_first).item())

        # dx_total = _delta(xT, x0)      
        # dy_total = _delta(yT, y0)    
        # dyaw_total = _delta(thT, th0) 

        # total_forward = abs(float(x0.item()))    
        # total_lateral = abs(float(y0.item()))
        # net_yaw_change = abs(float(th0.item()))

        # vel = features["vel"].squeeze(0)  # (2,)
        # acc = features["acc"].squeeze(0)  # (2,)
        # speed_mps = float(torch.linalg.norm(vel).item()) 
        # acc_abs   = float(torch.linalg.norm(acc).item()) 

        context_dict = {
                    "cmd_onehot": features["driving_command"].to(device, dtype=weight_dtype),
                    "vel":        features["vel"].to(device, dtype=weight_dtype),
                    "acc":        features["acc"].to(device, dtype=weight_dtype),
                }
        for reference_name in (
            "level_zero_ego_reference",
            "level_zero_ego_reference_mask",
            "level_zero_other_references",
            "level_zero_other_reference_mask",
        ):
            if reference_name in features:
                reference = features[reference_name].to(device)
                if torch.is_floating_point(reference):
                    reference = reference.to(dtype=weight_dtype)
                context_dict[reference_name] = reference

        level_k_kwargs = {}
        if self.level_k_config.enabled:
            required_agent_features = (
                "agent_current_states",
                "agent_types",
                "agent_mask",
            )
            missing_agent_features = [
                name
                for name in required_agent_features
                if name not in features
            ]
            if missing_agent_features:
                raise ValueError(
                    "Level-k forward_test is missing current annotation "
                    "features: " + ", ".join(missing_agent_features)
                )
            level_k_kwargs = {
                "level_k_config": self.level_k_config,
                "ego_velocity": features["vel"].to(
                    device,
                    dtype=weight_dtype,
                ),
                "other_agent_current_states": features[
                    "agent_current_states"
                ].to(device, dtype=weight_dtype),
                "other_agent_types": features["agent_types"].to(device),
                "other_agent_mask": features["agent_mask"].to(device),
            }
            for reference_name in (
                "level_zero_ego_reference",
                "level_zero_ego_reference_mask",
                "level_zero_other_references",
                "level_zero_other_reference_mask",
            ):
                if reference_name in features:
                    level_k_kwargs[reference_name] = context_dict[
                        reference_name
                    ]

        prompt = (
            f"A high-quality, photorealistic dashboard camera view of autonomous driving. "
            f"Based on the past 2 seconds videos, "
            f"predict and generate the next 4 seconds of realistic driving continuation, "
            f"Maintain temporal consistency, stable camera perspective, natural motion flow without jitter or artifacts, "
            f"clear details, and realistic physics. "
        )
        

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cond_pils: List[Image.Image] = self._tensor_video_to_pils(cond_tensor)
        torch.cuda.synchronize()
        timing['image_preprocessing'] = time.perf_counter() - t0
        

        # height = int(getattr(args, "height", H_in))
        # width  = int(getattr(args, "width",  W_in))
        # height, width = self.round_to_vae_resolution(height, width)

        num_frames = int(getattr(args, "num_frames", T_cond))
        num_steps  = int(getattr(args, "num_inference_step", 20))
        seed       = int(getattr(args, "seed", 42))
        negative_prompt = getattr(args, "negative_prompt",
            "worst quality, low quality, blurry, jittery, distorted, motion blur, ghosting, "
            "flickering, stuttering, camera shake, unstable footage, warping, trailing artifacts, "
            "temporal inconsistency, jerky motion, choppy framerate"
        )
        
        action_chunk = int(getattr(args, "action_chunk",8))  
        action_dim   = int(getattr(args, "action_dim", 3))                         

        if self.view_mode == "surround6":
            height, width = 256, 3072
        else:
            height, width = 768, 1344
            if hasattr(args, "height") and args.height:
                height = int(args.height)
            if hasattr(args, "width") and args.width:
                width = int(args.width)

        condition = LTXVideoCondition(video=cond_pils, frame_index=0)
        generator = torch.Generator(device=device).manual_seed(seed)

       
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = self.pipe.infer(
            conditions=[condition],
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=10,
            guidance_scale=1.0,
            height=height,
            width=width,
            generator=generator,
            num_frames=18,
            output_type="latent", 
            return_action=True,
            return_video=False,
            action_chunk=action_chunk,   
            action_dim=action_dim,
            context_dict=context_dict,
            noise_seed=42,
            pixel_wise_timestep=True,
            image_cond_noise_scale=0.0,
            **level_k_kwargs,
        )
        torch.cuda.synchronize()
        timing['pipe_infer'] = time.perf_counter() - t1

        
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        preds = out.frames if hasattr(out, "frames") else out
        actions = preds["actions"] if isinstance(preds, dict) else preds[0]["action"]  # (B, action_chunk, action_dim)
        final_action_clip_value = 1.0
        if final_action_clip_value is not None:
            actions.clamp_(-final_action_clip_value, final_action_clip_value)
        actions = denorm_odo(actions)

        actions = actions[0].detach().float().cpu()  # (action_chunk, action_dim)
        torch.cuda.synchronize()
        timing['action_postprocessing'] = time.perf_counter() - t2
        
        
        self.last_inference_timing = timing
        
        
        result: Dict[str, object] = {
            "actions": actions.numpy(),  
            "actions_tensor": actions,   
        }
        if isinstance(preds, dict) and "level_k_output" in preds:
            result["level_k_output"] = preds["level_k_output"]
        return result

    def compute_trajectory(
        self,
        agent_input: AgentInput,
        scene=None,
    ) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param agent_input: Dataclass with causal sensor and Ego inputs.
        :param scene: Required only by Oracle Level-k to read annotations at
            current_idx. Frames after current_idx are never accessed here.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        
        
        timing = {}
        
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        if self.level_k_config.enabled:
            if not self.level_k_config.oracle_agent_states:
                raise NotImplementedError(
                    "Level-k inference currently requires explicit NAVSIM "
                    "annotations."
                )
            if scene is None:
                raise ValueError(
                    "Level-k inference requires the current NAVSIM scene."
                )
            features.update(
                self._surround_agent_builder.compute_current_inputs(scene)
            )
        # Do not read future targets during inference. Target builders are
        # training-only; calling any of them here would unnecessarily access
        # future Ego states, images, or annotations.
        torch.cuda.synchronize()
        timing['feature_building'] = time.perf_counter() - t0

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        # forward pass
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["actions"]
        torch.cuda.synchronize()
        timing['forward_pass'] = time.perf_counter() - t1
        
        
        if hasattr(self, 'last_inference_timing'):
            timing.update(self.last_inference_timing)
        
        
        self.last_inference_timing = timing
        
        # extract trajectory
        return Trajectory(poses)

    def compute_trajectory_vis(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        if self.level_k_config.enabled:
            raise ValueError(
                "compute_trajectory_vis requires a scene when Level-k is enabled."
            )

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["actions"]
        return Trajectory(poses)


    @staticmethod
    def _decode_paths_from_tensor(path_tensor: torch.Tensor) -> List[str]:
        """
        Decodes a batch of path tensors back into a list of file path strings.
        
        Args:
            path_tensor (torch.Tensor): A 2D tensor of shape 
                (batch_size, max_path_length) from the collate_fn.
        
        Returns:
            List[str]: A list of decoded file path strings.
        """
        decoded_paths = []
        for single_path_tensor in path_tensor:
            chars = []
            for code in single_path_tensor:
                code_item = code.item()
                if code_item == 0: 
                    break
                chars.append(chr(code_item))
            decoded_paths.append("".join(chars))
        return decoded_paths

    @staticmethod
    def _compute_rel_from_T44(T: torch.Tensor):
            if not torch.is_tensor(T):
                T = torch.from_numpy(T)
            T = T.to(dtype=torch.float32)
            device = T.device
            n = T.shape[0]
            poses = torch.zeros(n, 2, dtype=T.dtype, device=device)
            yaws  = torch.zeros(n,     dtype=T.dtype, device=device)

            for i in range(1, n):
                T_prev_inv = torch.linalg.inv(T[i-1])
                T_rel = T_prev_inv @ T[i]          # (4,4)
                poses[i] = T_rel[:2, 3]
                R = T_rel[:3, :3]
                yaws[i] = torch.atan2(R[1, 0], R[0, 0])
            return poses, yaws

    @staticmethod
    def _tensor_video_to_pils(video: torch.Tensor):
        """
        video: (T,C,H,W) in [-1,1] 或 [0,1]
        return: List[PIL.Image]
        """
        x = video.detach().cpu()
        if x.min() < 0:
            x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)
        pils = []
        T = x.shape[0]
        for t in range(T):
            arr = (x[t].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            pils.append(Image.fromarray(arr))
        return pils

def pil_frames_to_video(pil_frames, output_path, fps):
    try:
        export_to_video(pil_frames, output_path, fps=fps)
        return True
    except Exception as e:
        print(f"❌ Error saving video {output_path}: {e}")
        return False

def denorm_odo(normalized_trajectory: torch.Tensor) -> torch.Tensor:
    """Denormalizes trajectory from [-1, 1] back to original coordinate space."""
    x = (normalized_trajectory[..., 0:1] + 1) / 2 * 66.74 - 1.57
    y = (normalized_trajectory[..., 1:2] + 1) / 2 * 42 - 19.68
    heading = (normalized_trajectory[..., 2:3] + 1) / 2 * 3.53 - 1.67
    return torch.cat([x, y, heading], dim=-1)

def _to_np(x):
    import numpy as np
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def print_pred_gt_together(poses, gt, precision=3, max_rows=None):
    import numpy as np
    p = _to_np(poses)
    g = _to_np(gt)

    if p.ndim == 1: p = p[:, None]
    if g.ndim == 1: g = g[:, None]

    T = min(len(p), len(g))
    if T == 0:
        print("No overlapping timesteps (T=0).")
        return

    p, g = p[:T], g[:T]
    D = min(p.shape[1], g.shape[1])
    p, g = p[:, :D], g[:, :D]

    err = np.linalg.norm(p - g, axis=1)
    np.set_printoptions(precision=precision, suppress=True)

    print(f"{'t':>3} | {'pred':>20} | {'gt':>20} | {'L2':>8}")
    print("-" * 64)
    rows = T if max_rows is None else min(max_rows, T)
    for i in range(rows):
        print(f"{i:>3} | {p[i]} | {g[i]} | {err[i]:>8.{precision}f}")
    if rows < T:
        print(f"... ({T-rows} more rows)")
    print("-" * 64)
    print(f"Avg L2 over T={T}: {err.mean():.{precision}f}")
