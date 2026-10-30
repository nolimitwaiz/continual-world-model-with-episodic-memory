#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing G0 attestation outside a Slurm allocation." >&2
  exit 2
fi

if [[ $# -ne 2 ]]; then
  echo "Usage: attest_g0.sbatch RESOLVED_CONFIG.yaml OUTPUT.json" >&2
  exit 2
fi

: "${CMWM_DATA_ROOT:?CMWM_DATA_ROOT must point to cluster data storage}"
: "${CMWM_RUN_ROOT:?CMWM_RUN_ROOT must point to cluster run storage}"

CONFIG_PATH="$1"
OUTPUT_PATH="$2"
if [[ ! -r "$CONFIG_PATH" ]]; then
  echo "Configuration is not readable: $CONFIG_PATH" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${CMWM_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${CMWM_PYTHON:-python}"

mkdir -p "$(dirname -- "$OUTPUT_PATH")"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

if ! "$PYTHON_BIN" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "Refusing G0 attestation without a CUDA device visible to PyTorch." >&2
  exit 3
fi

exec "$PYTHON_BIN" -m cmwm.cli attest-g0 \
  --config "$CONFIG_PATH" \
  --output "$OUTPUT_PATH"
