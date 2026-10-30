#!/usr/bin/env python3
"""Audit whether the full validation pool can support a balanced endpoint."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cmwm.config import load_config
from cmwm.evaluation.conformance import PRIMARY_LAGS, PRIMARY_TASKS, effective_query_lag
from cmwm.infrastructure.artifacts import atomic_write_json
from cmwm.schemas import DatasetSplit
from cmwm.synthetic.generator import SyntheticEpisodeGenerator
from cmwm.synthetic.manifest import content_sha256
from cmwm.synthetic.queries import generate_queries


TARGET_PER_CELL = 64


def audit_validation_pool(config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    generator = SyntheticEpisodeGenerator(
        base_seed=int(config["data"]["split_seed"]),
        episode_steps=int(config["data"]["max_episode_steps"]),
        schedule_policy=str(config["data"].get("generator_policy", "random_v1")),
    )
    episode_count = int(config["data"]["validation_episodes"])
    counts: Counter[tuple[str, int]] = Counter()
    task_counts: Counter[str] = Counter()
    query_ids: set[str] = set()
    duplicate_query_ids: set[str] = set()
    required_candidates: list[dict[str, object]] = []

    for episode_index in range(episode_count):
        episode = generator.generate_episode(DatasetSplit.VALIDATION, episode_index)
        for example in generate_queries(episode, lags=config["data"]["query_lags"]):
            query_id = example.query.query_id
            if query_id in query_ids:
                duplicate_query_ids.add(query_id)
            query_ids.add(query_id)
            task = example.query.kind.value
            lag = effective_query_lag(episode, example)
            counts[(task, lag)] += 1
            task_counts[task] += 1
            if task in PRIMARY_TASKS and lag in PRIMARY_LAGS:
                required_candidates.append(
                    {
                        "episode_index": episode_index,
                        "episode_id": episode.episode_id,
                        "query_id": query_id,
                        "task": task,
                        "effective_lag": lag,
                    }
                )

    required_candidates.sort(
        key=lambda row: (
            str(row["task"]),
            int(row["effective_lag"]),
            str(row["episode_id"]),
            str(row["query_id"]),
        )
    )
    required_cells = {
        f"{task}@{lag}": counts[(task, lag)]
        for task in PRIMARY_TASKS
        for lag in PRIMARY_LAGS
    }
    insufficient_cells = {
        key: count
        for key, count in required_cells.items()
        if count < TARGET_PER_CELL
    }
    lag_histograms = {
        task: {
            str(lag): count
            for (observed_task, lag), count in sorted(counts.items())
            if observed_task == task
        }
        for task in PRIMARY_TASKS
    }
    return {
        "artifact_type": "cmwm_validation_pool_endpoint_capacity_v1",
        "claim_eligible": False,
        "split": DatasetSplit.VALIDATION.value,
        "test_accessed": False,
        "validation_episode_count": episode_count,
        "target_per_task_lag_cell": TARGET_PER_CELL,
        "target_balanced_cohort_size": (
            len(PRIMARY_TASKS) * len(PRIMARY_LAGS) * TARGET_PER_CELL
        ),
        "unique_query_count": len(query_ids),
        "duplicate_query_ids": sorted(duplicate_query_ids),
        "task_counts": dict(sorted(task_counts.items())),
        "effective_lag_histograms": lag_histograms,
        "required_cells": required_cells,
        "insufficient_cells": insufficient_cells,
        "balanced_manifest_constructible": not insufficient_cells,
        "required_candidate_identity_sha256": content_sha256(required_candidates),
        "decision": (
            "freeze_condition_blind_balanced_validation_manifest"
            if not insufficient_cells
            else "repair_query_and_episode_construction_before_checkpoint_evaluation"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite pool audit: {args.output}")
    report = audit_validation_pool(args.config)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
