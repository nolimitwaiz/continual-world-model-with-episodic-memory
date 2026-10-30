#!/usr/bin/env bash
#SBATCH --job-name=cmwm-sherlock-bm25-v2
#SBATCH --partition=cpu
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing Sherlock evaluation outside a Slurm allocation." >&2
  exit 2
fi
: "${CMWM_PROJECT_ROOT:?CMWM_PROJECT_ROOT must point to the staged project}"
: "${CMWM_CHEN_SOURCE_PROJECT:?CMWM_CHEN_SOURCE_PROJECT must point to the existing Chen workspace}"
: "${CMWM_RUN_ROOT:?CMWM_RUN_ROOT must point to cluster run storage}"
: "${CMWM_PYTHON:?CMWM_PYTHON must point to the locked environment Python}"

OUTPUT_ROOT="${CMWM_RUN_ROOT}/sherlock_graph_free_bm25_v2/${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_ROOT}"
cd "${CMWM_PROJECT_ROOT}"
export PYTHONPATH="${CMWM_PROJECT_ROOT}/src"

"${CMWM_PYTHON}" scripts/run_sherlock_graph_free.py \
  --config configs/chen/sherlock_graph_free_bm25_v2.json \
  --source-project "${CMWM_CHEN_SOURCE_PROJECT}" \
  --output-directory "${OUTPUT_ROOT}" \
  --split development \
  > "${OUTPUT_ROOT}/run.stdout.txt" 2> "${OUTPUT_ROOT}/run.stderr.txt"
find "${OUTPUT_ROOT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum \
  > "${OUTPUT_ROOT}/SHA256SUMS.txt"
