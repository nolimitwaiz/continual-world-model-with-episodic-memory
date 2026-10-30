#!/usr/bin/env bash
#SBATCH --job-name=cmwm-resume-compare
#SBATCH --partition=cpu
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing checkpoint comparison outside a Slurm allocation." >&2
  exit 2
fi

: "${CMWM_PROJECT_ROOT:?CMWM_PROJECT_ROOT must point to the staged project}"
: "${CMWM_PYTHON:?CMWM_PYTHON must point to the locked environment Python}"
: "${CMWM_REFERENCE_CHECKPOINT:?CMWM_REFERENCE_CHECKPOINT is required}"
: "${CMWM_RESUMED_CHECKPOINT:?CMWM_RESUMED_CHECKPOINT is required}"
: "${CMWM_COMPARE_OUTPUT:?CMWM_COMPARE_OUTPUT is required}"

mkdir -p "$(dirname -- "${CMWM_COMPARE_OUTPUT}")"
cd "${CMWM_PROJECT_ROOT}"
"${CMWM_PYTHON}" scripts/compare_resume_checkpoints.py \
  "${CMWM_REFERENCE_CHECKPOINT}" \
  "${CMWM_RESUMED_CHECKPOINT}" \
  > "${CMWM_COMPARE_OUTPUT}"
sha256sum "${CMWM_COMPARE_OUTPUT}" > "${CMWM_COMPARE_OUTPUT}.sha256"
