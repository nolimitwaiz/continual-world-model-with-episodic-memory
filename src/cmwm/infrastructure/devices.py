"""Fail-closed enforcement for official Prototype 1 runs."""

from __future__ import annotations

import os
from typing import Mapping


class ClusterExecutionError(RuntimeError):
    pass


def require_cluster_cuda(environ: Mapping[str, str] | None = None) -> str:
    """Return the Slurm job ID or reject execution outside a CUDA allocation."""

    environment = os.environ if environ is None else environ
    job_id = environment.get("SLURM_JOB_ID") or environment.get("SLURM_JOBID")
    if not job_id:
        raise ClusterExecutionError(
            "training/evaluation is disabled outside a Slurm allocation"
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - cluster environment failure
        raise ClusterExecutionError("PyTorch is unavailable in the Slurm job") from exc
    if not torch.cuda.is_available():  # pragma: no cover - requires cluster hardware
        raise ClusterExecutionError("the Slurm job has no CUDA device visible to PyTorch")
    return str(job_id)
