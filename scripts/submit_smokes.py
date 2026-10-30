#!/usr/bin/env python3
"""Submit serial, fail-closed CMWM engineering smoke jobs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cmwm.config import load_config  # noqa: E402
from cmwm.smoke import SMOKE_SUITES  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/prototype1/smoke.yaml",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=SMOKE_SUITES,
        default=list(SMOKE_SUITES),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-submit", action="store_true")
    parser.add_argument("--account", default=os.environ.get("CMWM_SLURM_ACCOUNT"))
    parser.add_argument("--partition", default=os.environ.get("CMWM_SLURM_PARTITION"))
    parser.add_argument("--qos", default=os.environ.get("CMWM_SLURM_QOS"))
    parser.add_argument("--time", default="00:20:00")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--memory", default="16G")
    return parser


def _missing_scheduler(args: argparse.Namespace) -> list[str]:
    return [name for name in ("account", "partition", "qos") if not getattr(args, name)]


def submit(args: argparse.Namespace) -> dict[str, object]:
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    payload: dict[str, object] = {
        "config": str(config_path),
        "experiment": config["experiment"]["name"],
        "suites": list(args.suites),
        "scheduler": {
            "account": args.account,
            "partition": args.partition,
            "qos": args.qos,
            "time": args.time,
            "gpus": args.gpus,
            "cpus_per_task": args.cpus_per_task,
            "memory": args.memory,
        },
    }
    if args.dry_run:
        payload["dry_run"] = True
        payload["would_invoke_sbatch"] = False
        return payload

    if not args.confirm_submit:
        raise ValueError("real smoke submission requires --confirm-submit")
    if os.environ.get("CMWM_SUBMISSION_AUTHORIZED") != "explicit-user-authorization":
        raise ValueError("CMWM_SUBMISSION_AUTHORIZED does not record explicit authorization")
    missing = _missing_scheduler(args)
    if missing:
        raise ValueError("missing scheduler arguments: " + ", ".join(missing))
    if not os.environ.get("CMWM_DATA_ROOT") or not os.environ.get("CMWM_RUN_ROOT"):
        raise ValueError("CMWM_DATA_ROOT and CMWM_RUN_ROOT are required")
    if args.gpus <= 0 or args.cpus_per_task <= 0:
        raise ValueError("GPU and CPU counts must be positive")

    script = PROJECT_ROOT / "cluster/slurm/smoke_single.sbatch"
    run_root = Path(os.environ["CMWM_RUN_ROOT"]).expanduser().resolve()
    log_dir = run_root / "slurm"
    log_dir.mkdir(parents=True, exist_ok=True)
    job_ids: dict[str, str] = {}
    dependency: str | None = None
    for suite in args.suites:
        command = [
            "sbatch",
            "--parsable",
            f"--account={args.account}",
            f"--partition={args.partition}",
            f"--qos={args.qos}",
            f"--gres=gpu:{args.gpus}",
            f"--cpus-per-task={args.cpus_per_task}",
            f"--mem={args.memory}",
            f"--time={args.time}",
            f"--job-name=cmwm_smoke_{suite}",
            f"--output={log_dir}/%x_%j.log",
            f"--error={log_dir}/%x_%j.log",
            "--export=ALL",
        ]
        if dependency is not None:
            command.append(f"--dependency=afterok:{dependency}")
        command.extend((str(script), str(config_path), suite))
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        job_id = completed.stdout.strip().split(";", 1)[0]
        if not job_id:
            raise RuntimeError(f"empty Slurm job ID for suite {suite}")
        job_ids[suite] = job_id
        dependency = job_id
    payload["submitted"] = True
    payload["jobs"] = job_ids
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(json.dumps(submit(args), indent=2, sort_keys=True))
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"smoke submission failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
