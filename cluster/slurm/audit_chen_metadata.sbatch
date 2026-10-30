#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing Chen metadata audit outside a Slurm allocation." >&2
  exit 2
fi

if [[ $# -ne 1 ]]; then
  echo "Usage: audit_chen_metadata.sbatch OUTPUT.json" >&2
  exit 2
fi

: "${CMWM_RUN_ROOT:?CMWM_RUN_ROOT must point to cluster run storage}"

OUTPUT_PATH="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${CMWM_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${CMWM_PYTHON:-python}"

mkdir -p "$(dirname -- "$OUTPUT_PATH")"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m pytest -q tests/unit/test_openneuro_metadata_audit.py
exec "$PYTHON_BIN" scripts/audit_openneuro_metadata.py --output "$OUTPUT_PATH"
