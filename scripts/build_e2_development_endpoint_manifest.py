#!/usr/bin/env python3
"""Freeze the balanced CMWM-Synth-E2 validation endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cmwm.config import load_config
from cmwm.evaluation.conformance import PRIMARY_LAGS, PRIMARY_TASKS
from cmwm.evaluation.endpoint_manifest import (
    EndpointCandidate,
    build_balanced_endpoint_manifest,
)
from cmwm.g0 import dataset_source_sha256
from cmwm.infrastructure.artifacts import atomic_write_json
from cmwm.schemas import DatasetSplit
from cmwm.synthetic.generator import SyntheticEpisodeGenerator
from cmwm.synthetic.manifest import content_sha256
from cmwm.synthetic.queries import effective_query_lag, generate_queries


def build_manifest(config_path: Path, *, examples_per_cell: int) -> dict[str, object]:
    config = load_config(config_path)
    policy = str(config["data"].get("generator_policy", "random_v1"))
    if policy != "exact_task_lag_v2":
        raise ValueError("E2 endpoint manifest requires exact_task_lag_v2")
    generator = SyntheticEpisodeGenerator(
        base_seed=int(config["data"]["split_seed"]),
        episode_steps=int(config["data"]["max_episode_steps"]),
        schedule_policy=policy,
    )
    candidates: list[EndpointCandidate] = []
    episode_count = int(config["data"]["validation_episodes"])
    for episode_index in range(episode_count):
        episode = generator.generate_episode(DatasetSplit.VALIDATION, episode_index)
        for example in generate_queries(episode, lags=config["data"]["query_lags"]):
            lag = effective_query_lag(episode, example)
            if example.query.kind.value not in PRIMARY_TASKS or lag not in PRIMARY_LAGS:
                continue
            candidates.append(
                EndpointCandidate(
                    episode_index=episode_index,
                    episode_id=episode.episode_id,
                    query_id=example.query.query_id,
                    task=example.query.kind.value,
                    effective_lag=lag,
                    oracle_answer_sha256=content_sha256(example.oracle_answer),
                )
            )
    candidates.sort(
        key=lambda candidate: (
            candidate.task,
            candidate.effective_lag,
            candidate.episode_id,
            candidate.query_id,
        )
    )
    return build_balanced_endpoint_manifest(
        candidates,
        tasks=PRIMARY_TASKS,
        lags=PRIMARY_LAGS,
        examples_per_cell=examples_per_cell,
        dataset_source_sha256=dataset_source_sha256(),
        config_sha256=content_sha256(config),
        generator_policy=policy,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--examples-per-cell", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite endpoint manifest: {args.output}")
    manifest = build_manifest(args.config, examples_per_cell=args.examples_per_cell)
    atomic_write_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
