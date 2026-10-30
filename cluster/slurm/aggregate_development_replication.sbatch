#!/usr/bin/env bash
#SBATCH --job-name=cmwm-repl-aggregate
#SBATCH --partition=cpu
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing replication aggregation outside a Slurm allocation." >&2
  exit 2
fi
: "${CMWM_PROJECT_ROOT:?CMWM_PROJECT_ROOT must point to the staged project}"
: "${CMWM_RUN_ROOT:?CMWM_RUN_ROOT must point to the replication run root}"
: "${CMWM_PYTHON:?CMWM_PYTHON must point to the locked environment Python}"

OUTPUT_ROOT="${CMWM_RUN_ROOT}/development_replication_summary/${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_ROOT}"
cd "${CMWM_PROJECT_ROOT}"
export PYTHONPATH="${CMWM_PROJECT_ROOT}/src"

"${CMWM_PYTHON}" scripts/aggregate_development_replication.py \
  --run-root "${CMWM_RUN_ROOT}" \
  --output "${OUTPUT_ROOT}/summary.json" \
  > "${OUTPUT_ROOT}/run.stdout.txt" 2> "${OUTPUT_ROOT}/run.stderr.txt"
find "${OUTPUT_ROOT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum \
  > "${OUTPUT_ROOT}/SHA256SUMS.txt"
