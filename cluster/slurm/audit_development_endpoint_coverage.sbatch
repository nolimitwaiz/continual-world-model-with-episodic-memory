#!/usr/bin/env bash
#SBATCH --job-name=cmwm-endpoint-audit
#SBATCH --partition=cpu
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing endpoint audit outside a Slurm allocation." >&2
  exit 2
fi
: "${CMWM_PROJECT_ROOT:?CMWM_PROJECT_ROOT must point to the staged project}"
: "${CMWM_REPLICATION_ROOT:?CMWM_REPLICATION_ROOT must point to completed replication runs}"
: "${CMWM_RUN_ROOT:?CMWM_RUN_ROOT must point to cluster run storage}"
: "${CMWM_PYTHON:?CMWM_PYTHON must point to the locked environment Python}"

OUTPUT_ROOT="${CMWM_RUN_ROOT}/development_endpoint_coverage/${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_ROOT}"
cd "${CMWM_PROJECT_ROOT}"
export PYTHONPATH="${CMWM_PROJECT_ROOT}/src"

"${CMWM_PYTHON}" -m pytest -q tests/unit/test_evaluation_conformance.py \
  > "${OUTPUT_ROOT}/tests.stdout.txt" 2> "${OUTPUT_ROOT}/tests.stderr.txt"
"${CMWM_PYTHON}" scripts/audit_development_endpoint_coverage.py \
  --run-root "${CMWM_REPLICATION_ROOT}" \
  --output "${OUTPUT_ROOT}/coverage.json" \
  > "${OUTPUT_ROOT}/audit.stdout.txt" 2> "${OUTPUT_ROOT}/audit.stderr.txt"
find "${OUTPUT_ROOT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum \
  > "${OUTPUT_ROOT}/SHA256SUMS.txt"
