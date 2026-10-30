#!/usr/bin/env bash
#SBATCH --job-name=cmwm-e3-manifest
#SBATCH --partition=cpu
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing E3 endpoint construction outside a Slurm allocation." >&2
  exit 2
fi
: "${CMWM_PROJECT_ROOT:?CMWM_PROJECT_ROOT must point to the staged project}"
: "${CMWM_RUN_ROOT:?CMWM_RUN_ROOT must point to cluster run storage}"
: "${CMWM_PYTHON:?CMWM_PYTHON must point to the locked environment Python}"
: "${CMWM_E3_CONFIG:?CMWM_E3_CONFIG must point to the frozen E3 config}"

OUTPUT_ROOT="${CMWM_RUN_ROOT}/e3_development_endpoint_manifest/${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_ROOT}"
cd "${CMWM_PROJECT_ROOT}"
export PYTHONPATH="${CMWM_PROJECT_ROOT}/src"

"${CMWM_PYTHON}" -m pytest -q tests/unit/test_endpoint_manifest.py \
  > "${OUTPUT_ROOT}/tests.stdout.txt" 2> "${OUTPUT_ROOT}/tests.stderr.txt"
"${CMWM_PYTHON}" scripts/build_e3_development_endpoint_manifest.py \
  --config "${CMWM_E3_CONFIG}" \
  --examples-per-cell 64 \
  --output "${OUTPUT_ROOT}/manifest.json" \
  > "${OUTPUT_ROOT}/build.stdout.txt" 2> "${OUTPUT_ROOT}/build.stderr.txt"
find "${OUTPUT_ROOT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum \
  > "${OUTPUT_ROOT}/SHA256SUMS.txt"
