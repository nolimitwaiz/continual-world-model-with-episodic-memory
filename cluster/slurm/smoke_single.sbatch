#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing smoke execution outside a Slurm allocation." >&2
  exit 2
fi

if [[ $# -ne 2 ]]; then
  echo "Usage: smoke_single.sbatch RESOLVED_CONFIG.yaml SUITE" >&2
  exit 2
fi

: "${CMWM_DATA_ROOT:?CMWM_DATA_ROOT must point to cluster data storage}"
: "${CMWM_RUN_ROOT:?CMWM_RUN_ROOT must point to cluster run storage}"

CONFIG_PATH="$1"
SUITE="$2"
case "$SUITE" in
  environment|generator|components|g0-readiness) ;;
  *) echo "Unknown smoke suite: $SUITE" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${CMWM_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${CMWM_PYTHON:-python}"
RESULT_DIR="$CMWM_RUN_ROOT/engineering_smoke"
RESULT_PATH="$RESULT_DIR/${SLURM_JOB_ID}_${SUITE}.json"

mkdir -p "$RESULT_DIR"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

if ! "$PYTHON_BIN" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "Refusing smoke execution without a CUDA device visible to PyTorch." >&2
  exit 3
fi

exec "$PYTHON_BIN" -m cmwm.cli smoke \
  --config "$CONFIG_PATH" \
  --suite "$SUITE" \
  --output "$RESULT_PATH"
