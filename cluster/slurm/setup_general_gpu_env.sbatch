#!/usr/bin/env bash
#SBATCH --job-name=cmwm-env-cu121
#SBATCH --partition=cpu
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing environment setup outside a Slurm allocation." >&2
  exit 2
fi

: "${CMWM_BASE_PYTHON:?CMWM_BASE_PYTHON is required}"
: "${CMWM_GENERAL_ENV:?CMWM_GENERAL_ENV is required}"
: "${CMWM_ENV_LOG_DIR:?CMWM_ENV_LOG_DIR is required}"

mkdir -p "${CMWM_ENV_LOG_DIR}"
if [[ ! -x "${CMWM_GENERAL_ENV}/bin/python" ]]; then
  "${CMWM_BASE_PYTHON}" -m venv "${CMWM_GENERAL_ENV}"
fi

"${CMWM_GENERAL_ENV}/bin/python" -m pip install --upgrade "pip>=24,<26"
"${CMWM_GENERAL_ENV}/bin/python" -m pip install \
  torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
"${CMWM_GENERAL_ENV}/bin/python" -m pip install \
  "numpy>=1.26,<3" \
  "PyYAML>=6.0.2,<7" \
  "pytest>=8.3,<9" \
  "pytest-cov>=6,<7" \
  "ruff>=0.9,<1"

"${CMWM_GENERAL_ENV}/bin/python" -m pip freeze \
  > "${CMWM_ENV_LOG_DIR}/requirements.lock.txt"
"${CMWM_GENERAL_ENV}/bin/python" -VV \
  > "${CMWM_ENV_LOG_DIR}/python-version.txt" 2>&1
"${CMWM_GENERAL_ENV}/bin/python" -c \
  'import json, torch; print(json.dumps({"torch": torch.__version__, "compiled_cuda": torch.version.cuda}, sort_keys=True))' \
  > "${CMWM_ENV_LOG_DIR}/torch-build.json"
sha256sum \
  "${CMWM_ENV_LOG_DIR}/requirements.lock.txt" \
  "${CMWM_ENV_LOG_DIR}/python-version.txt" \
  "${CMWM_ENV_LOG_DIR}/torch-build.json" \
  > "${CMWM_ENV_LOG_DIR}/SHA256SUMS.txt"

