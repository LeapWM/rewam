#!/usr/bin/env bash
set -euo pipefail

# Run one Unified Level-1 sample and render trajectory/BEV/Video-DiT
# diagnostics.  Usage:
#   bash scripts/visualization/run_unified_token_diagnostics.sh [TOKEN]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ENV_BIN="${DRIVELAW_ENV_BIN:-/root/miniconda3/envs/drivelaw/bin}"
TOKEN="${1:-${DRIVELAW_DIAGNOSTIC_TOKEN:-0004474e9e3f5470}}"

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${PROJ}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/c2-worldmodel/training_data/OpenScene/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/dahuafs/userdata/2639638/navsim_exp}"
export PYTHONPATH="${PROJ}:${PYTHONPATH:-}"

: "${DRIVELAW_VIDEO_MODEL_ROOT:?Set DRIVELAW_VIDEO_MODEL_ROOT to the Video-DiT weights directory}"
: "${DRIVELAW_DIFFUSION_WEIGHTS:?Set DRIVELAW_DIFFUSION_WEIGHTS to the ReWAM checkpoint}"
export DRIVELAW_VIDEO_MODEL_ROOT DRIVELAW_DIFFUSION_WEIGHTS
export DRIVELAW_MODEL_CONFIG="${DRIVELAW_MODEL_CONFIG:-${PROJ}/navsim/agents/videodrive/configs/ltx_model/video_model_infer_navsim.yaml}"

export DRIVELAW_LEVEL_K_ENABLED="${DRIVELAW_LEVEL_K_ENABLED:-true}"
export DRIVELAW_ORACLE_AGENT_STATES="${DRIVELAW_ORACLE_AGENT_STATES:-true}"
export DRIVELAW_REASONING_LEVEL="${DRIVELAW_REASONING_LEVEL:-1}"

PY_SITE="$("${ENV_BIN}/python" -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${PY_SITE}/nvidia/nvjitlink/lib:${PY_SITE}/nvidia/cusparse/lib:${PY_SITE}/nvidia/cusolver/lib:${PY_SITE}/nvidia/cublas/lib:${PY_SITE}/nvidia/cudnn/lib:${PY_SITE}/nvidia/nccl/lib:${PY_SITE}/nvidia/cufft/lib:${PY_SITE}/nvidia/curand/lib:${PY_SITE}/nvidia/cuda_runtime/lib:/root/miniconda3/envs/drivelaw/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

VIS_SCRIPT="${SCRIPT_DIR}/visualize_videodrive_prediction.py"
EVALUATION_EXPERIMENT="${DRIVELAW_EVALUATION_EXPERIMENT:-videodrive_agent_eval_unified_k1_step100000_ema_full}"
LATEST_EVALUATION_POINTER="${NAVSIM_EXP_ROOT}/${EVALUATION_EXPERIMENT}.latest_run.txt"

if [[ ! -x "${ENV_BIN}/python" ]]; then
    echo "[FAIL] Python executable not found: ${ENV_BIN}/python" >&2
    exit 2
fi
if [[ ! -f "${VIS_SCRIPT}" ]]; then
    echo "[FAIL] Diagnostic Python script not found: ${VIS_SCRIPT}" >&2
    exit 2
fi
if [[ ! -f "${DRIVELAW_MODEL_CONFIG}" ]]; then
    echo "[FAIL] Model config not found: ${DRIVELAW_MODEL_CONFIG}" >&2
    exit 2
fi
if [[ ! -f "${DRIVELAW_DIFFUSION_WEIGHTS}" ]]; then
    echo "[FAIL] Unified checkpoint not found: ${DRIVELAW_DIFFUSION_WEIGHTS}" >&2
    exit 2
fi
if [[ ! -d "${OPENSCENE_DATA_ROOT}/navsim_logs/test" ]]; then
    echo "[FAIL] NAVSIM test logs not found under ${OPENSCENE_DATA_ROOT}" >&2
    exit 2
fi

EVAL_CSV="${DRIVELAW_EVALUATION_CSV:-}"
if [[ -z "${EVAL_CSV}" ]]; then
    if [[ ! -f "${LATEST_EVALUATION_POINTER}" ]]; then
        echo "[FAIL] Evaluation run pointer not found: ${LATEST_EVALUATION_POINTER}" >&2
        exit 2
    fi
    EVAL_RUN_DIR="$(<"${LATEST_EVALUATION_POINTER}")"
    EVAL_CSV="$(find "${EVAL_RUN_DIR}" \
        -maxdepth 1 \
        -type f \
        -name '*.csv' \
        ! -name '*.tmp' \
        -printf '%T@ %p\n' \
        | sort -nr \
        | sed -n '1s/^[^ ]* //p')"
fi
if [[ -z "${EVAL_CSV}" || ! -f "${EVAL_CSV}" ]]; then
    echo "[FAIL] Unified evaluation CSV not found: ${EVAL_CSV:-<empty>}" >&2
    exit 2
fi

STAMP="$(date +%Y.%m.%d.%H.%M.%S)"
OUTPUT_DIR="${DRIVELAW_DIAGNOSTIC_OUTPUT_DIR:-${NAVSIM_EXP_ROOT}/unified_token_diagnostics/${TOKEN}/${STAMP}}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${NAVSIM_EXP_ROOT}/unified_token_diagnostics"
printf '%s\n' "${OUTPUT_DIR}" > \
    "${NAVSIM_EXP_ROOT}/unified_token_diagnostics/${TOKEN}.latest_run.txt"

echo "[INFO] Unified token diagnostics"
echo "[INFO] token=${TOKEN}"
echo "[INFO] project=${PROJ}"
echo "[INFO] checkpoint=${DRIVELAW_DIFFUSION_WEIGHTS}"
echo "[INFO] evaluation_csv=${EVAL_CSV}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] output_dir=${OUTPUT_DIR}"

set +e
"${ENV_BIN}/python" "${VIS_SCRIPT}" \
    --token "${TOKEN}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-config "${DRIVELAW_MODEL_CONFIG}" \
    --data-root "${OPENSCENE_DATA_ROOT}" \
    --data-split test \
    --evaluation-csv "${EVAL_CSV}" \
    --device cuda \
    2>&1 | tee "${OUTPUT_DIR}/run.log"
STATUS="${PIPESTATUS[0]}"
set -e

echo "[INFO] inference exit code=${STATUS}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
if [[ "${STATUS}" -eq 0 ]]; then
    find "${OUTPUT_DIR}" -maxdepth 2 -type f -printf '%p  %k KB\n' | sort
else
    echo "[FAIL] Unified token diagnostics failed; inspect ${OUTPUT_DIR}/run.log" >&2
fi
exit "${STATUS}"
