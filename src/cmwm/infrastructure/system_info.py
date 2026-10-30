"""Capture enough scheduler and accelerator metadata to audit a run."""

from __future__ import annotations

import os
import platform
import socket
from typing import Any


def collect_system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOBID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count(),
                "gpu_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
    except ImportError:
        info["torch"] = None
    return info
