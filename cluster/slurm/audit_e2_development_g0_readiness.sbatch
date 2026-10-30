#!/usr/bin/env bash
#SBATCH --job-name=cmwm-e2-g0-ready
#SBATCH --partition=cpu
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing E2 G0 readiness outside a Slurm allocation." >&2
  exit 2
fi
: "${CMWM_PROJECT_ROOT:?CMWM_PROJECT_ROOT must point to the staged project}"
: "${CMWM_RUN_ROOT:?CMWM_RUN_ROOT must point to cluster run storage}"
: "${CMWM_PYTHON:?CMWM_PYTHON must point to the locked environment Python}"
: "${CMWM_E2_CONFIG:?CMWM_E2_CONFIG must point to the frozen E2 config}"

OUTPUT_ROOT="${CMWM_RUN_ROOT}/e2_development_g0_readiness/${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_ROOT}"
cd "${CMWM_PROJECT_ROOT}"
export PYTHONPATH="${CMWM_PROJECT_ROOT}/src"

EXTRA_ARGS=()
if [[ -n "${CMWM_G0_TRAIN_EPISODES:-}" ]]; then
  EXTRA_ARGS+=(--train-episodes "${CMWM_G0_TRAIN_EPISODES}")
fi
if [[ -n "${CMWM_G0_VALIDATION_EPISODES:-}" ]]; then
  EXTRA_ARGS+=(--validation-episodes "${CMWM_G0_VALIDATION_EPISODES}")
fi
if [[ "${CMWM_G0_REQUIRE_PASS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--require-pass)
fi
"${CMWM_PYTHON}" scripts/audit_e2_development_g0_readiness.py \
  --config "${CMWM_E2_CONFIG}" \
  "${EXTRA_ARGS[@]}" \
  --output "${OUTPUT_ROOT}/readiness.json" \
  > "${OUTPUT_ROOT}/audit.stdout.txt" 2> "${OUTPUT_ROOT}/audit.stderr.txt"
find "${OUTPUT_ROOT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum \
  > "${OUTPUT_ROOT}/SHA256SUMS.txt"
