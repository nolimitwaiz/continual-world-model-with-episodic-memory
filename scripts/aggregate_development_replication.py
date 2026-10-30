#!/usr/bin/env python3
"""Aggregate a complete five-seed development matrix without test access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cmwm.evaluation.replication import aggregate_development_replication
from cmwm.infrastructure.artifacts import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite replication summary: {args.output}")
    report = aggregate_development_replication(args.run_root)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
