#!/usr/bin/env python3
"""Validate, snapshot, and submit Prototype 1 jobs without invoking a shell."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cmwm.config import dump_config, load_config  # noqa: E402
from cmwm.infrastructure.artifacts import atomic_write_text  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--confirm-submit",
        action="store_true",
        help="required for a real submission after explicit user authorization",
    )
    parser.add_argument("--account", default=os.environ.get("CMWM_SLURM_ACCOUNT"))
    parser.add_argument("--partition", default=os.environ.get("CMWM_SLURM_PARTITION"))
    parser.add_argument("--qos", default=os.environ.get("CMWM_SLURM_QOS"))
    parser.add_argument(
        "--nodelist",
        default=os.environ.get("CMWM_SLURM_NODELIST"),
        help="optional scheduler node constraint after a successful compatibility probe",
    )
    parser.add_argument("--time", default="01:00:00")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--memory", default="32G")
    parser.add_argument("--array-max-parallel", type=int, default=4)
    parser.add_argument(
        "--seeds",
        help="optional comma-separated seed expansion applied to every selected config",
    )
    return parser


def _manifest_paths(path: Path) -> list[Path]:
    manifest = path.expanduser().resolve()
    if not manifest.is_file():
        raise ValueError(f"manifest does not exist: {manifest}")
    result: list[Path] = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        result.append(candidate if candidate.is_absolute() else manifest.parent / candidate)
    if not result:
        raise ValueError("manifest contains no configuration paths")
    return result


def _parse_seeds(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return ()
    try:
        seeds = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("--seeds must be a comma-separated list of integers") from error
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique non-negative integers")
    return seeds


def _expand_seeds(
    configs: list[tuple[Path, dict]], seeds: tuple[int, ...]
) -> list[tuple[Path, dict]]:
    if not seeds:
        return configs
    expanded: list[tuple[Path, dict]] = []
    for path, config in configs:
        for seed in seeds:
            value = deepcopy(config)
            value["experiment"]["seed"] = seed
            expanded.append((path, value))
    return expanded


def _validated_configs(args: argparse.Namespace) -> list[tuple[Path, dict]]:
    paths = [args.config] if args.config is not None else _manifest_paths(args.manifest)
    configs = [(path.expanduser().resolve(), load_config(path)) for path in paths]
    return _expand_seeds(configs, _parse_seeds(args.seeds))


def _require_scheduler_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("account", "partition")
        if not getattr(args, name)
    ]
    if missing:
        joined = ", ".join(f"--{name}" for name in missing)
        raise ValueError(f"missing scheduler arguments: {joined}")
    if args.gpus <= 0 or args.cpus_per_task <= 0 or args.array_max_parallel <= 0:
        raise ValueError("GPU, CPU, and array concurrency values must be positive")


def _common_sbatch(args: argparse.Namespace, log_pattern: Path) -> list[str]:
    command = [
        "sbatch",
        "--parsable",
        f"--account={args.account}",
        f"--partition={args.partition}",
        f"--gres=gpu:{args.gpus}",
        f"--cpus-per-task={args.cpus_per_task}",
        f"--mem={args.memory}",
        f"--time={args.time}",
        f"--output={log_pattern}",
        f"--error={log_pattern}",
        "--export=ALL",
    ]
    if args.qos:
        command.insert(4, f"--qos={args.qos}")
    if args.nodelist:
        command.insert(4, f"--nodelist={args.nodelist}")
    return command


def _dry_run_payload(
    configs: list[tuple[Path, dict]], args: argparse.Namespace
) -> dict[str, object]:
    return {
        "dry_run": True,
        "configs": [str(path) for path, _ in configs],
        "experiments": [config["experiment"]["name"] for _, config in configs],
        "seeds": [int(config["experiment"]["seed"]) for _, config in configs],
        "requires_development_endpoint_manifest": any(
            config["data"].get("generator_policy")
            in {"exact_task_lag_v2", "confusable_events_v3"}
            for _, config in configs
        ),
        "scheduler": {
            "account": args.account,
            "partition": args.partition,
            "qos": args.qos,
            "nodelist": args.nodelist,
            "time": args.time,
            "gpus": args.gpus,
            "cpus_per_task": args.cpus_per_task,
            "memory": args.memory,
        },
    }


def submit(args: argparse.Namespace) -> str:
    configs = _validated_configs(args)
    if args.dry_run:
        return json.dumps(_dry_run_payload(configs, args), indent=2, sort_keys=True)

    if not args.confirm_submit:
        raise ValueError(
            "real submission refused: pass --confirm-submit only after explicit user authorization"
        )
    if os.environ.get("CMWM_SUBMISSION_AUTHORIZED") != "explicit-user-authorization":
        raise ValueError(
            "real submission refused: CMWM_SUBMISSION_AUTHORIZED must record "
            "explicit user authorization"
        )

    _require_scheduler_args(args)
    run_root_value = os.environ.get("CMWM_RUN_ROOT")
    data_root_value = os.environ.get("CMWM_DATA_ROOT")
    if not run_root_value or not data_root_value:
        raise ValueError("CMWM_RUN_ROOT and CMWM_DATA_ROOT are required for submission")
    g0_attestation = os.environ.get("CMWM_G0_ATTESTATION")
    if not g0_attestation or not Path(g0_attestation).expanduser().is_file():
        raise ValueError("CMWM_G0_ATTESTATION must point to a readable G0 artifact")
    requires_endpoint = any(
        config["data"].get("generator_policy")
        in {"exact_task_lag_v2", "confusable_events_v3"}
        for _, config in configs
    )
    endpoint_manifest = os.environ.get("CMWM_DEVELOPMENT_ENDPOINT_MANIFEST")
    if requires_endpoint and (
        not endpoint_manifest
        or not Path(endpoint_manifest).expanduser().is_file()
    ):
        raise ValueError(
            "CMWM_DEVELOPMENT_ENDPOINT_MANIFEST must point to the sealed development endpoint"
        )
    run_root = Path(run_root_value).expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    submission_dir = run_root / "_submissions" / timestamp
    log_dir = run_root / "slurm"
    submission_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=True)

    resolved_paths: list[Path] = []
    for index, (_, config) in enumerate(configs):
        name = config["experiment"]["name"]
        resolved = submission_dir / f"{index:03d}_{name}.yaml"
        atomic_write_text(resolved, dump_config(config))
        resolved_paths.append(resolved)

    if len(resolved_paths) == 1:
        script = PROJECT_ROOT / "cluster/slurm/train_single.sbatch"
        log_pattern = log_dir / "%x_%j.log"
        command = _common_sbatch(args, log_pattern) + [
            f"--job-name={configs[0][1]['experiment']['name']}",
            str(script),
            str(resolved_paths[0]),
        ]
    else:
        manifest = submission_dir / "resolved_manifest.txt"
        atomic_write_text(manifest, "".join(f"{path}\n" for path in resolved_paths))
        script = PROJECT_ROOT / "cluster/slurm/sweep_array.sbatch"
        log_pattern = log_dir / "%x_%A_%a.log"
        command = _common_sbatch(args, log_pattern) + [
            "--job-name=cmwm_p1_array",
            f"--array=0-{len(resolved_paths) - 1}%{args.array_max_parallel}",
            str(script),
            str(manifest),
        ]

    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(submit(args))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"submission failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
