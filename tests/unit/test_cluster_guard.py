from __future__ import annotations

import json
from pathlib import Path

import pytest

from cmwm.infrastructure.devices import ClusterExecutionError, require_cluster_cuda
from cmwm.training import record_training_failure


def test_cluster_guard_rejects_non_slurm_environment() -> None:
    with pytest.raises(ClusterExecutionError, match="Slurm"):
        require_cluster_cuda({})


def test_cuda_entrypoints_configure_deterministic_cublas() -> None:
    project_root = Path(__file__).resolve().parents[2]
    for name in (
        "attest_g0.sbatch",
        "smoke_single.sbatch",
        "train_single.sbatch",
        "sweep_array.sbatch",
    ):
        script = (project_root / "cluster" / "slurm" / name).read_text()
        assert 'CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"' in script


def test_runtime_failure_writes_terminal_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CMWM_TEST_RUN_ROOT", str(tmp_path))
    monkeypatch.setenv("SLURM_JOB_ID", "test-job-17")
    config = {
        "paths": {"run_root_env": "CMWM_TEST_RUN_ROOT"},
        "experiment": {"name": "failure-fixture", "seed": 3},
    }

    record_training_failure(config, RuntimeError("deliberate fixture failure"))

    status_path = tmp_path / "failure-fixture" / "seed-3" / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["exception_type"] == "RuntimeError"
    assert status["exception_message"] == "deliberate fixture failure"
    assert status["slurm_job_id"] == "test-job-17"
