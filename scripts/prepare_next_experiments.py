#!/usr/bin/env python3
"""Print an inert draft inventory. No data reads, training, network or submission.

This intentionally has no execute flag or Slurm interface. Passing configuration
booleans that claim authorization causes an error rather than launching anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cmwm.evaluation.confirmation import CONDITIONS, FAMILIES, LAGS
from cmwm.evaluation.conformance import PRIMARY_TASKS
from cmwm.evaluation.replication import EXPECTED_SEEDS


def preparation_inventory(config: dict) -> dict:
    if config.get("artifact_type") != "cmwm_next_stage_preparation_v1":
        raise ValueError("unknown preparation schema")
    for flag in ("protocol_frozen", "execution_authorized", "test_access_authorized", "training_authorized"):
        if config.get(flag) is not False:
            raise PermissionError(f"preparation requires {flag}=false")
    synthetic = config["synthetic_confirmation"]
    for key, expected in (("conditions", CONDITIONS), ("seeds", EXPECTED_SEEDS),
                          ("families", FAMILIES), ("lags", LAGS), ("tasks", PRIMARY_TASKS)):
        if tuple(synthetic[key]) != tuple(expected):
            raise ValueError(f"unexpected synthetic {key}")
    if synthetic["checkpoint_step"] != 1000 or synthetic["examples_per_cell"] != 64:
        raise ValueError("draft endpoint and final-checkpoint policy must not silently change")
    if synthetic["primary_baseline"] != "fixed_chunks" or synthetic["mandatory_secondary_baseline"] != "random_matched":
        raise ValueError("baseline selection cannot be changed by this preparation command")
    sherlock = config["sherlock_dense_development"]
    if sherlock["active_split"] != "development" or sherlock["sealed_test_authorized"] is not False:
        raise PermissionError("Sherlock test remains closed")
    if sherlock["records_per_condition"] * sherlock["fixed_size"] != sherlock["atomic_rows"]:
        raise ValueError("fixed groups do not match the common record count")
    if not synthetic["blockers"] or not sherlock["blockers"]:
        raise ValueError("unresolved readiness work cannot be hidden")
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return {
        "status": "prepared_not_executable",
        "plan_content_sha256": hashlib.sha256(canonical).hexdigest(),
        "training_jobs": 0,
        "submitted_jobs": 0,
        "proposed_synthetic_checkpoint_evaluations": len(CONDITIONS) * len(EXPECTED_SEEDS),
        "proposed_unique_synthetic_queries": 2 * 4 * 2 * synthetic["examples_per_cell"],
        "proposed_sherlock_groupings": 2 + len(sherlock["random_seeds"]),
        "blockers": {
            "synthetic": synthetic["blockers"], "sherlock": sherlock["blockers"],
        },
        "execution": "No dataset, checkpoint, encoder, network, or scheduler is accessed by this command.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1]
                        / "configs/next_stage/preparation.json")
    args = parser.parse_args()
    print(json.dumps(preparation_inventory(json.loads(args.config.read_text())), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
