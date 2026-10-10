# ReWAM: Reciprocal World Action Model

This repository is the inference-only release of **ReWAM: Reciprocal World Action Models for Interactive Autonomous Driving**. It contains the Level-K planner, checkpoint loading, NAVSIM evaluation, submission export, and qualitative visualization code required to reproduce inference results.

Training pipelines, auxiliary-loss implementations, cache builders, mining scripts, gradient calibration, and experiment logs are intentionally excluded.

![ReWAM framework](assets/readme/rewam_framework.png)

## Scope

The released inference path includes:

- DriveLaW video/action backbone integration
- Level-K reciprocal ego/agent reasoning
- strategy conditioning and Other Action Expert inference
- NAVSIM feature construction and trajectory decoding
- distributed NAVTEST PDM scoring
- submission export and visualization utilities
- inference-focused unit tests

The reference checkpoint uses:

| Setting | Value |
| --- | --- |
| Reasoning depth | Level-1 |
| Other Action DiT | 28 layers |
| Prediction horizon | 8 steps |
| Sampling interval | 0.5 s |
| Agent state source | current-frame oracle states |
| Reported NAVTEST PDMS | 90.6426 |

A checkpoint trained with another reasoning depth or Other Action DiT depth is not architecture-compatible with this configuration.

## Repository Layout

```text
.
|-- assets/readme/                         # Paper and qualitative figures
|-- ReWAM-main/
|   |-- Action-DiT/
|   |   |-- navsim/agents/videodrive/      # ReWAM agent and model code
|   |   |-- navsim/planning/script/        # PDM scoring and submission entrypoints
|   |   |-- scripts/                       # Inference and visualization helpers
|   |   `-- tests/                         # Retained inference tests
|   `-- Video-DiT/Infer/                   # Standalone world-model inference
`-- README.md
```

## Installation

Create the project environment and install the action-planning package:

```bash
cd ReWAM-main/Action-DiT
conda env create -f environment.yml
conda activate drivelaw
pip install -e .
```

The modified `diffusers` source under `ReWAM-main/Video-DiT/Infer/diffusers/src` is used by standalone video inference. NAVSIM planning uses the environment's installed dependencies.

## Required Assets

The checkpoint is not stored in Git. Prepare:

1. NAVSIM/OpenScene data and nuPlan maps.
2. NAVSIM metric cache for NAVTEST.
3. Video-DiT pretrained model directory.
4. ReWAM action checkpoint `diffusion_pytorch_model.safetensors`.

Set the runtime paths:

```bash
export REWAM_ROOT=/path/to/ReWAM__Reciprocal_World_Action_Model
export ACT_ROOT="$REWAM_ROOT/ReWAM-main/Action-DiT"
export OPENSCENE_DATA_ROOT=/path/to/OpenScene/dataset
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_EXP_ROOT=/path/to/navsim_exp

export DRIVELAW_VIDEO_MODEL_ROOT=/path/to/Video-DiT
export DRIVELAW_DIFFUSION_WEIGHTS=/path/to/diffusion_pytorch_model.safetensors
export DRIVELAW_MODEL_CONFIG="$ACT_ROOT/navsim/agents/videodrive/configs/ltx_model/video_model_infer_navsim_other28.yaml"

export DRIVELAW_LEVEL_K_ENABLED=true
export DRIVELAW_ORACLE_AGENT_STATES=true
export DRIVELAW_REASONING_LEVEL=1
export DRIVELAW_MAX_REASONING_LEVEL=3
export PYTHONPATH="$ACT_ROOT"
```

The checkpoint used to verify this release has SHA-256:

```text
e259ca91aa373bbf666a508a841faa557ed4541186dbd37050f490c3d01c94b7
```

## NAVTEST Evaluation

Run distributed evaluation from `Action-DiT`:

```bash
cd "$ACT_ROOT"

torchrun --standalone --nproc_per_node=8 \
  navsim/planning/script/run_pdm_score_videodrive.py \
  agent=videodrive_agent \
  train_test_split=navtest \
  metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache" \
  experiment_name=rewam_navtest \
  output_dir="$NAVSIM_EXP_ROOT/rewam_navtest"
```

For a one-token smoke test, append:

```bash
'train_test_split.scene_filter.tokens=[TOKEN]'
```

Each rank writes a CSV shard containing validity, inference time, NC, DAC, EP, TTC, Comfort, DDC, and PDMS. Merge shards with the retained evaluation utilities before reporting the full NAVTEST score.


## Verification

Run the retained inference test suite:

```bash
cd "$ACT_ROOT"
PYTHONDONTWRITEBYTECODE=1 pytest -q
```

Release verification completed with:

```text
298 passed
```

A real one-token NAVTEST smoke test loaded all 219 checkpoint tensors with no missing, unexpected, or mismatched keys and completed trajectory inference plus PDM scoring successfully.

## Qualitative Results

<table width="100%">
  <tr>
    <td align="center" valign="top" width="33%">
      <img src="assets/readme/narrow_road.png" width="220"><br>
      <b>Narrow road</b>
    </td>
    <td align="center" valign="top" width="33%">
      <img src="assets/readme/left_turn.png" width="220"><br>
      <b>Left turn</b>
    </td>
    <td align="center" valign="top" width="33%">
      <img src="assets/readme/going_straight.png" width="220"><br>
      <b>Going straight</b>
    </td>
  </tr>
</table>

## Notes

- The released checkpoint and YAML must be kept architecture-compatible.
- Metric-cache corruption is an evaluation-data issue; rebuild invalid zero-byte or truncated cache files.
- Small per-token score changes can occur when diffusion sampling uses a different random stream. Full-set reporting must use a fixed seed and the same distributed setup.
- Training and fine-tuning are outside the scope of this repository.
