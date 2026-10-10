#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_PYTHON="/root/miniconda3/envs/drivelaw/bin/python"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${DEFAULT_PYTHON}" ]]; then
    PYTHON_BIN="${DEFAULT_PYTHON}"
  else
    PYTHON_BIN="python"
  fi
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/visualization/analyze_navtest_performance_by_interaction.py "$@"
