#!/usr/bin/env bash
#SBATCH --job-name=cmwm-unit-cpu
#SBATCH --partition=cpu
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing cluster unit tests outside a Slurm allocation." >&2
  exit 2
fi
: "${CMWM_PROJECT_ROOT:?CMWM_PROJECT_ROOT must point to the staged project}"
: "${CMWM_RUN_ROOT:?CMWM_RUN_ROOT must point to cluster run storage}"
: "${CMWM_PYTHON:?CMWM_PYTHON must point to the locked environment Python}"

RUN_ROOT="${CMWM_RUN_ROOT}/unit_tests_cpu/${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"
cd "${CMWM_PROJECT_ROOT}"
export PYTHONPATH="${CMWM_PROJECT_ROOT}/src"

"${CMWM_PYTHON}" -m pytest -q tests/unit \
  --junitxml="${RUN_ROOT}/unit_tests.xml" \
  > "${RUN_ROOT}/tests.stdout.txt" 2> "${RUN_ROOT}/tests.stderr.txt"
"${CMWM_PYTHON}" -VV > "${RUN_ROOT}/python-version.txt" 2>&1
find "${RUN_ROOT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum \
  > "${RUN_ROOT}/SHA256SUMS.txt"
