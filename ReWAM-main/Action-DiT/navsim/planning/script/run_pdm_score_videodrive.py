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
from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import math
import warnings
import lzma
import pickle
import os
import uuid
import time
import torch
import torch.distributed as dist
import pickle
import io
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
import numpy as np

from nuplan.planning.script.builders.logging_builder import build_logger
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.videodrive.level_k.config import LevelKConfig
from navsim.agents.videodrive.level_k.inference_metadata import (
    build_inference_metadata,
    write_inference_metadata,
)
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    EgoAreaIndex,
)

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in cast",
    category=RuntimeWarning,
    module=r"nuplan\.common\.maps\.nuplan_map\.utils",
)

logger = logging.getLogger(__name__)


def _allowed_eval_logs() -> set[str] | None:
    """Load an optional fixed log allow-list for leakage-safe NAVTRAIN runs."""

    path_text = os.environ.get("DRIVELAW_EVAL_ALLOWED_LOGS_FILE", "").strip()
    if not path_text:
        return None
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation log allow-list does not exist: {path}")
    logs = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not logs:
        raise ValueError(f"Evaluation log allow-list is empty: {path}")
    return logs


CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=False
    )
    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    tokens_to_evaluate = sorted(tokens_to_evaluate) 
    # Use DistributedSampler to split the tokens for each GPU
    
    # Distribute tasks to different GPUs by iterating over the sampled tokens
    pdm_results: List[Dict[str, Any]] = []
    inference_times: List[float] = []
    skipped_existing_candidates = 0
    
    for idx, (token) in enumerate(tokens_to_evaluate):
        if dist.get_rank() == 0:
            logger.info(f"Rank {dist.get_rank()} processing scenario {idx+1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}")

        score_row: Dict[str, Any] = {
            "token": token,
            "log_name": log_names[0],
            "valid": True,
        }
        inference_time = None
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            agent_input = scene_loader.get_agent_input_from_token(token)
            requires_scene = agent.requires_scene
            if requires_scene:
                scene = scene_loader.get_scene_from_token(token)
            
            torch.cuda.synchronize()  
            start_time = time.perf_counter()
            
            if requires_scene:
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input)
            
            torch.cuda.synchronize()  
            end_time = time.perf_counter()
            inference_time = end_time - start_time  
            inference_times.append(inference_time)
            
            timing_info = getattr(agent, 'last_inference_timing', {})
            
            timing_str = f"total={inference_time:.4f}s"
            if timing_info:
                parts = []
                if 'feature_building' in timing_info:
                    parts.append(f"feature_building={timing_info['feature_building']:.4f}s")
                if 'image_preprocessing' in timing_info:
                    parts.append(f"image_preprocessing={timing_info['image_preprocessing']:.4f}s")
                if 'pipe_infer' in timing_info:
                    parts.append(f"pipe_infer={timing_info['pipe_infer']:.4f}s")
                if 'action_postprocessing' in timing_info:
                    parts.append(f"action_postprocessing={timing_info['action_postprocessing']:.4f}s")
                if 'forward_pass' in timing_info:
                    parts.append(f"forward_pass={timing_info['forward_pass']:.4f}s")
                if parts:
                    timing_str += " [" + ", ".join(parts) + "]"
            
            logger.info(f"[Rank {dist.get_rank()}] Token {token}: inference time = {timing_str}")
            
            score_row['inference_time_breakdown'] = timing_info
            
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))
            pred_idx = 1
            offroad = scorer._ego_areas[
                pred_idx, :, EgoAreaIndex.NON_DRIVABLE_AREA
            ]
            violation_indices = np.flatnonzero(offroad)
            interval = float(scorer.proposal_sampling.interval_length)
            score_row.update(
                {
                    "drivable_violation_frames": int(offroad.sum()),
                    "drivable_first_violation_time_s": (
                        float(violation_indices[0]) * interval
                        if len(violation_indices)
                        else math.inf
                    ),
                    "ttc_infraction_time_s": float(
                        scorer.time_to_ttc_infraction(pred_idx)
                    ),
                    "collision_time_s": float(
                        scorer.time_to_at_fault_collision(pred_idx)
                    ),
                }
            )
            score_row['rank'] = dist.get_rank()
            score_row['inference_time'] = inference_time
            score_row["skipped_existing_candidate"] = False
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False
            score_row['inference_time'] = None
            score_row['inference_time_breakdown'] = {}
            score_row["skipped_existing_candidate"] = False

        pdm_results.append(score_row)
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(tokens_to_evaluate):
            if inference_times:
                current_avg = sum(inference_times) / len(inference_times)
                logger.info(f"[Rank {dist.get_rank()}] Progress: {idx+1}/{len(tokens_to_evaluate)} scenarios processed. Current average inference time: {current_avg:.4f} seconds ({len(inference_times)} valid trajectories, {skipped_existing_candidates} skipped existing candidates)")
            elif skipped_existing_candidates:
                logger.info(f"[Rank {dist.get_rank()}] Progress: {idx+1}/{len(tokens_to_evaluate)} scenarios processed. Skipped {skipped_existing_candidates} existing hard-negative candidates.")
    
    if inference_times:
        avg_inference_time = sum(inference_times) / len(inference_times)
        logger.info(f"[Rank {dist.get_rank()}] Final average inference time per trajectory: {avg_inference_time:.4f} seconds ({len(inference_times)} valid trajectories)")
    if skipped_existing_candidates:
        logger.info(
            "[Rank %s] Reused %s existing hard-negative candidates.",
            dist.get_rank(),
            skipped_existing_candidates,
        )
    serialized_score_rows = pickle.dumps(pdm_results)
    return serialized_score_rows

def broadcast_object(obj: Any, device: torch.device, src: int = 0) -> Any:
    """
    Helper function to broadcast an object from the source rank to all other processes.
    :param obj: Object to broadcast.
    :param device: Device to use for tensor operations.
    :param src: Source rank.
    :return: Broadcasted object.
    """
    if dist.get_rank() == src:
        buffer = pickle.dumps(obj)
        tensor = torch.ByteTensor(list(buffer)).to(device)
        size_tensor = torch.tensor(len(tensor)).to(device)
        dist.broadcast(size_tensor, src=src)
        dist.broadcast(tensor, src=src)
    else:
        size_tensor = torch.tensor(0).to(device)
        dist.broadcast(size_tensor, src=src)
        tensor = torch.ByteTensor(size_tensor.item()).to(device)
        dist.broadcast(tensor, src=src)
        buffer = tensor.cpu().numpy().tobytes()
        obj = pickle.loads(buffer)
    return obj


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
    )
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')  

    build_logger(cfg)
    logger.info(
        "Skipping WorkerPool construction; torchrun ranks process PDM "
        "scoring shards directly."
    )
    metadata_agent = instantiate(cfg.agent)
    experiment_metadata = build_inference_metadata(
        getattr(metadata_agent, "level_k_config", LevelKConfig())
    )
    logger.info(
        "Experiment identity: %s | variant=%s | oracle=%s",
        experiment_metadata["model_name"],
        experiment_metadata["evaluation_variant"],
        experiment_metadata["oracle_agent_states"],
    )

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    # scene_loader = SceneLoader(
    #     sensor_blobs_path=None,
    #     data_path=Path(cfg.navsim_log_path),
    #     scene_filter=instantiate(cfg.train_test_split.scene_filter),
    #     sensor_config=SensorConfig.build_no_sensors(),
    # )
    # metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    # tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    scene_loader = SceneLoader(
            sensor_blobs_path=None,
            data_path=Path(cfg.navsim_log_path),
            scene_filter=instantiate(cfg.train_test_split.scene_filter),
            sensor_config=SensorConfig.build_no_sensors(),
        )
    if rank == 0:
        metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
        allowed_logs = _allowed_eval_logs()
        scene_tokens = set(scene_loader.tokens)
        if allowed_logs is not None:
            available_logs = set(scene_loader.token_to_log_file.values())
            effective_logs = allowed_logs.intersection(available_logs)
            expected_logs_text = os.environ.get(
                "DRIVELAW_EVAL_EXPECTED_ALLOWED_LOGS", ""
            ).strip()
            if expected_logs_text and len(effective_logs) != int(expected_logs_text):
                raise ValueError(
                    "Evaluation allow-list intersection has an unexpected size: "
                    f"expected={expected_logs_text}, actual={len(effective_logs)}, "
                    f"configured={len(allowed_logs)}, available={len(available_logs)}"
                )
            if not effective_logs:
                raise ValueError(
                    "Evaluation allow-list has no logs in the selected split."
                )
            scene_tokens = {
                token
                for token in scene_tokens
                if scene_loader.token_to_log_file[token] in effective_logs
            }
            logger.info(
                "Restricted evaluation to %d effective logs (%d configured) "
                "and %d scene tokens.",
                len(effective_logs),
                len(allowed_logs),
                len(scene_tokens),
            )
        tokens_to_evaluate = sorted(scene_tokens & set(metric_cache_loader.tokens))
        num_missing_metric_cache_tokens = len(scene_tokens - set(metric_cache_loader.tokens))
        num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - scene_tokens)
        if num_missing_metric_cache_tokens > 0:
            logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
        if num_unused_metric_cache_tokens > 0:
            logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    else:
        tokens_to_evaluate = []


    tokens_to_evaluate = broadcast_object(tokens_to_evaluate, device=device, src=0)


    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
    # data_points = [
    #     {
    #         "cfg": cfg,
    #         "log_file": log_file,
    #         "tokens": tokens_list,
    #     }
    #     for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    # ]
    sampler = InferenceSampler(len(tokens_to_evaluate))

    data_points = []
    for idx in sampler:
        token = tokens_to_evaluate[idx]
        log_file = scene_loader.token_to_log_file[token]  
        data_points.append({
            "cfg": cfg,
            "log_file": log_file,
            "tokens": [token],  
        })

    # Run PDM scoring and get serialized_score_rows
    serialized_score_rows = run_pdm_score(data_points)

    local_results = pickle.loads(serialized_score_rows)
    pdm_score_df = pd.DataFrame(local_results)
    if len(pdm_score_df) > 0:
        pdm_score_df["model_name"] = experiment_metadata["model_name"]
        pdm_score_df["comparison_family"] = experiment_metadata[
            "comparison_family"
        ]
        pdm_score_df["reasoning_level"] = experiment_metadata[
            "reasoning_level"
        ]
        pdm_score_df["oracle_agent_states"] = experiment_metadata[
            "oracle_agent_states"
        ]

    save_path = Path(cfg.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    result_path = save_path / f"{timestamp}.rank_{rank}.csv"
    pdm_score_df.to_csv(result_path)

    num_successful_scenarios = int(pdm_score_df["valid"].sum()) if "valid" in pdm_score_df else 0
    num_failed_scenarios = len(pdm_score_df) - num_successful_scenarios
    num_skipped_existing = (
        int(pdm_score_df["skipped_existing_candidate"].sum())
        if "skipped_existing_candidate" in pdm_score_df
        else 0
    )
    valid_inference_times = (
        pdm_score_df[pdm_score_df["valid"] == True]["inference_time"].dropna()
        if "inference_time" in pdm_score_df and "valid" in pdm_score_df
        else []
    )
    if len(valid_inference_times) > 0:
        logger.info(
            "[Rank %s] Average inference time for recomputed trajectories: %.4f seconds (%s valid trajectories)",
            rank,
            valid_inference_times.mean(),
            len(valid_inference_times),
        )

    if rank == 0:
        write_inference_metadata(
            save_path,
            experiment_metadata,
            artifact_type="pdm_evaluation",
            filename=f"{timestamp}.experiment_metadata.json",
            extra={
                "rank_result_pattern": f"{timestamp}.rank_*.csv",
                "aggregation": "per_rank_no_collective",
                "world_size": world_size,
            },
        )

    logger.info(
        "[Rank %s] Finished running evaluation shard. Successful: %s. Failed: %s. "
        "Skipped existing candidates: %s. Results are stored in: %s.",
        rank,
        num_successful_scenarios,
        num_failed_scenarios,
        num_skipped_existing,
        result_path,
    )
    dist.destroy_process_group()



if __name__ == "__main__":
    main()
