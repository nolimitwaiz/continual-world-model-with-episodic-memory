"""Command-line boundary between local-safe checks and cluster-only execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from cmwm.infrastructure.artifacts import atomic_write_json
from cmwm.infrastructure.devices import ClusterExecutionError, require_cluster_cuda


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cmwm")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-config")
    validate.add_argument("--config", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    attest = commands.add_parser("attest-g0")
    attest.add_argument("--config", type=Path, required=True)
    attest.add_argument("--output", type=Path, required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument(
        "--suite",
        choices=("environment", "generator", "components", "g0-readiness"),
        required=True,
    )
    smoke.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-config":
        from cmwm.config import load_config

        config = load_config(args.config)
        print(
            json.dumps(
                {
                    "valid": True,
                    "experiment": config["experiment"]["name"],
                    "run_class": config["experiment"]["run_class"],
                },
                sort_keys=True,
            )
        )
        return 0

    try:
        require_cluster_cuda()
    except ClusterExecutionError as exc:
        print(f"execution refused: {exc}", file=sys.stderr)
        return 2
    from cmwm.config import load_config

    config = load_config(args.config)
    if args.command == "attest-g0":
        from cmwm.g0 import build_g0_artifact

        result = build_g0_artifact(config, args.output)
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("passed") else 4
    if args.command == "smoke":
        from cmwm.smoke import run_smoke

        result = run_smoke(args.suite, config)
        if args.output is not None:
            atomic_write_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("passed") else 4
    if args.command == "train":
        from cmwm.training import (
            EngineeringInterruption,
            TrainingContractError,
            record_training_failure,
            train,
        )

        try:
            result = train(config)
        except TrainingContractError as exc:
            print(f"training refused: {exc}", file=sys.stderr)
            return 5
        except EngineeringInterruption as exc:
            print(f"training intentionally interrupted: {exc}", file=sys.stderr)
            return 75
        except Exception as exc:
            record_training_failure(config, exc)
            raise
        print(json.dumps(result, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
