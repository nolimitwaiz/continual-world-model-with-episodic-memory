#!/usr/bin/env bash
#SBATCH --job-name=cmwm-gpu-compat
#SBATCH --partition=gpu
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing GPU compatibility testing outside a Slurm allocation." >&2
  exit 2
fi

: "${CMWM_PYTHON:?CMWM_PYTHON is required}"
: "${CMWM_COMPAT_OUTPUT:?CMWM_COMPAT_OUTPUT is required}"

mkdir -p "$(dirname -- "${CMWM_COMPAT_OUTPUT}")"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
  --format=csv,noheader > "${CMWM_COMPAT_OUTPUT}.nvidia-smi.txt"
"${CMWM_PYTHON}" -c '
import json
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
device = torch.device("cuda")
x = torch.randn((64, 64), device=device, requires_grad=True)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = (x @ x.T).square().mean()
loss.backward()
torch.cuda.synchronize()
properties = torch.cuda.get_device_properties(device)
print(json.dumps({
    "artifact_type": "cmwm_general_gpu_compatibility",
    "passed": bool(torch.isfinite(loss).item() and torch.isfinite(x.grad).all().item()),
    "torch": torch.__version__,
    "compiled_cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(device),
    "compute_capability": [properties.major, properties.minor],
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "loss": float(loss.detach()),
}, indent=2, sort_keys=True))
' > "${CMWM_COMPAT_OUTPUT}"
sha256sum "${CMWM_COMPAT_OUTPUT}" "${CMWM_COMPAT_OUTPUT}.nvidia-smi.txt" \
  > "${CMWM_COMPAT_OUTPUT}.sha256"
