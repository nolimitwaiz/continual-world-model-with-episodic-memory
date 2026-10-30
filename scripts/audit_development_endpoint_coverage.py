#!/usr/bin/env python3
"""Regenerate frozen development cohorts and audit task/lag coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cmwm.evaluation.conformance import endpoint_rows, summarize_endpoint_rows
from cmwm.evaluation.replication import EXPECTED_RUNS, EXPECTED_SEEDS
from cmwm.infrastructure.artifacts import atomic_write_json
from cmwm.schemas import QueryKind
from cmwm.synthetic.generator import SyntheticEpisodeGenerator
from cmwm.training import _build_validation_cohort


def audit_development_endpoint_coverage(run_root: Path) -> dict[str, object]:
    root = run_root.expanduser().resolve()
    cohorts: dict[str, object] = {}
    for seed in EXPECTED_SEEDS:
        condition_hashes: dict[str, str] = {}
        reference_config: dict | None = None
        reference_result: dict | None = None
        for condition, run_name in EXPECTED_RUNS.items():
            run_directory = root / run_name / f"seed-{seed}"
            result_path = run_directory / "result.json"
            config_path = run_directory / "resolved_config.json"
            if not result_path.is_file() or not config_path.is_file():
                raise ValueError(f"missing completed development run: {run_directory}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if result.get("claim_eligible") is not False:
                raise ValueError(f"development result became claim eligible: {result_path}")
            if int(result.get("seed", -1)) != seed:
                raise ValueError(f"seed identity mismatch: {result_path}")
            observed_hash = result.get("validation_cohort_sha256")
            if not isinstance(observed_hash, str):
                raise ValueError(f"missing validation cohort identity: {result_path}")
            condition_hashes[condition] = observed_hash
            if reference_config is None:
                reference_config = config
                reference_result = result

        if len(set(condition_hashes.values())) != 1:
            raise ValueError(f"conditions are not paired within seed {seed}")
        assert reference_config is not None and reference_result is not None
        generator = SyntheticEpisodeGenerator(
            base_seed=int(reference_config["data"]["split_seed"]),
            episode_steps=int(reference_config["data"]["max_episode_steps"]),
        )
        cohort_seed = seed + 100_000
        if int(reference_result.get("validation_cohort_seed", -1)) != cohort_seed:
            raise ValueError(f"validation cohort seed mismatch for seed {seed}")
        cohort = _build_validation_cohort(
            generator=generator,
            config=reference_config,
            seed=cohort_seed,
        )
        if cohort.sha256 != next(iter(condition_hashes.values())):
            raise ValueError(f"regenerated validation cohort mismatch for seed {seed}")
        summary = summarize_endpoint_rows(endpoint_rows(cohort.batches))
        summary.pop("rows")
        cohorts[str(seed)] = {
            "cohort_seed": cohort.seed,
            "cohort_sha256": cohort.sha256,
            "condition_hashes": condition_hashes,
            **summary,
        }

    all_computable = all(
        bool(value["endpoint_computable"]) for value in cohorts.values()  # type: ignore[index]
    )
    return {
        "artifact_type": "cmwm_development_endpoint_coverage_audit_v1",
        "claim_eligible": False,
        "split": "validation",
        "test_accessed": False,
        "primary_tasks": [kind.value for kind in QueryKind],
        "primary_lags": [64, 128],
        "cohorts": cohorts,
        "endpoint_computable_for_all_seeds": all_computable,
        "decision": (
            "existing_cohorts_support_task_lag_evaluation"
            if all_computable
            else "stop_existing_cohort_evaluation_and_repair_query_coverage"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite coverage audit: {args.output}")
    report = audit_development_endpoint_coverage(args.run_root)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
