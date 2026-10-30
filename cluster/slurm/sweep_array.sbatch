#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Refusing to train outside a Slurm job array." >&2
  exit 2
fi

if [[ $# -ne 1 ]]; then
  echo "Usage: sweep_array.sbatch RESOLVED_MANIFEST.txt" >&2
  exit 2
fi

: "${CMWM_DATA_ROOT:?CMWM_DATA_ROOT must point to cluster data storage}"
: "${CMWM_RUN_ROOT:?CMWM_RUN_ROOT must point to cluster run storage}"

MANIFEST_PATH="$1"
if [[ ! -r "$MANIFEST_PATH" ]]; then
  echo "Manifest is not readable: $MANIFEST_PATH" >&2
  exit 2
fi

mapfile -t CONFIGS < <(awk 'NF && $1 !~ /^#/' "$MANIFEST_PATH")
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#CONFIGS[@]} )); then
  echo "Array index $SLURM_ARRAY_TASK_ID is outside manifest range." >&2
  exit 2
fi

CONFIG_PATH="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
if [[ ! -r "$CONFIG_PATH" ]]; then
  echo "Configuration is not readable: $CONFIG_PATH" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${CMWM_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${CMWM_PYTHON:-python}"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

if ! "$PYTHON_BIN" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "Refusing to train without a CUDA device visible to PyTorch." >&2
  exit 3
fi

exec "$PYTHON_BIN" -m cmwm.cli train --config "$CONFIG_PATH"
