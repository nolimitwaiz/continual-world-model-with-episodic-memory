#!/usr/bin/env python3
"""Compare uninterrupted and resumed checkpoints for exact training state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("resumed", type=Path)
    return parser


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def compare(reference: Path, resumed: Path) -> dict[str, Any]:
    first = torch.load(reference, map_location="cpu", weights_only=False)
    second = torch.load(resumed, map_location="cpu", weights_only=False)
    identity_fields = (
        "schema_version",
        "step",
        "max_steps",
        "condition",
        "seed",
        "config_sha256",
        "source_sha256",
        "g0_manifest_sha256",
        "validation_cohort_sha256",
    )
    checks = {
        "identity": all(first.get(key) == second.get(key) for key in identity_fields),
        "model": _equal(first.get("model"), second.get("model")),
        "optimizer": _equal(first.get("optimizer"), second.get("optimizer")),
        "sampler_rng_state": _equal(
            first.get("sampler_rng_state"), second.get("sampler_rng_state")
        ),
        "torch_rng_state": _equal(
            first.get("torch_rng_state"), second.get("torch_rng_state")
        ),
        "cuda_rng_state_all": _equal(
            first.get("cuda_rng_state_all"), second.get("cuda_rng_state_all")
        ),
        "latest_loss": _equal(first.get("latest_loss"), second.get("latest_loss")),
        "latest_evaluation": _equal(
            first.get("latest_evaluation"), second.get("latest_evaluation")
        ),
        "best_score": _equal(first.get("best_score"), second.get("best_score")),
        "best_step": _equal(first.get("best_step"), second.get("best_step")),
    }
    return {
        "artifact_type": "cmwm_exact_resume_comparison",
        "passed": all(checks.values()),
        "reference": str(reference.resolve()),
        "resumed": str(resumed.resolve()),
        "checks": checks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = compare(args.reference, args.resumed)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
