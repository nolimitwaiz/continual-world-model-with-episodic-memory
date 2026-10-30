#!/usr/bin/env bash
#SBATCH --job-name=cmwm-chen-ready
#SBATCH --partition=cpu
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing Chen readiness audit outside a Slurm allocation." >&2
  exit 2
fi
: "${CMWM_PROJECT_ROOT:?CMWM_PROJECT_ROOT must point to the staged project}"
: "${CMWM_CHEN_SOURCE_PROJECT:?CMWM_CHEN_SOURCE_PROJECT must point to the existing Chen workspace}"
: "${CMWM_RUN_ROOT:?CMWM_RUN_ROOT must point to cluster run storage}"
: "${CMWM_PYTHON:?CMWM_PYTHON must point to the locked environment Python}"

OUTPUT_ROOT="${CMWM_RUN_ROOT}/chen_behavioral_readiness/${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_ROOT}"
cd "${CMWM_PROJECT_ROOT}"

"${CMWM_PYTHON}" scripts/audit_chen_behavioral_readiness.py \
  --source-project "${CMWM_CHEN_SOURCE_PROJECT}" \
  --output "${OUTPUT_ROOT}/readiness.json" \
  > "${OUTPUT_ROOT}/audit.stdout.txt" 2> "${OUTPUT_ROOT}/audit.stderr.txt"
find "${OUTPUT_ROOT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum \
  > "${OUTPUT_ROOT}/SHA256SUMS.txt"
