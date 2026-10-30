#!/usr/bin/env bash
#SBATCH --job-name=cmwm-pool-capacity
#SBATCH --partition=cpu
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing validation-pool audit outside a Slurm allocation." >&2
  exit 2
fi
: "${CMWM_PROJECT_ROOT:?CMWM_PROJECT_ROOT must point to the staged project}"
: "${CMWM_REPLICATION_ROOT:?CMWM_REPLICATION_ROOT must point to completed replication runs}"
: "${CMWM_RUN_ROOT:?CMWM_RUN_ROOT must point to cluster run storage}"
: "${CMWM_PYTHON:?CMWM_PYTHON must point to the locked environment Python}"

OUTPUT_ROOT="${CMWM_RUN_ROOT}/validation_pool_endpoint_capacity/${SLURM_JOB_ID}"
CONFIG_PATH="${CMWM_CAPACITY_CONFIG:-${CMWM_REPLICATION_ROOT}/replication_c0_context/seed-11/resolved_config.json}"
if [[ ! -r "${CONFIG_PATH}" ]]; then
  echo "Capacity-audit configuration is unreadable: ${CONFIG_PATH}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
cd "${CMWM_PROJECT_ROOT}"
export PYTHONPATH="${CMWM_PROJECT_ROOT}/src"

"${CMWM_PYTHON}" scripts/audit_validation_pool_endpoint_capacity.py \
  --config "${CONFIG_PATH}" \
  --output "${OUTPUT_ROOT}/capacity.json" \
  > "${OUTPUT_ROOT}/audit.stdout.txt" 2> "${OUTPUT_ROOT}/audit.stderr.txt"
find "${OUTPUT_ROOT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum \
  > "${OUTPUT_ROOT}/SHA256SUMS.txt"
