#!/usr/bin/env python3
"""Freeze a condition-blind, task/lag-balanced E3 validation endpoint."""

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
from cmwm.synthetic.generator import SHORTCUT_RESISTANT_POLICY, SyntheticEpisodeGenerator
from cmwm.synthetic.manifest import content_sha256
from cmwm.synthetic.queries import effective_query_lag, generate_queries


E3_ENDPOINT_SALT = "cmwm-e3-development-endpoint-v1"


def build_manifest(config_path: Path, *, examples_per_cell: int) -> dict[str, object]:
    config = load_config(config_path)
    data = config["data"]
    policy = str(data.get("generator_policy", "random_v1"))
    if policy != SHORTCUT_RESISTANT_POLICY:
        raise ValueError(f"E3 endpoint requires {SHORTCUT_RESISTANT_POLICY}")
    if int(data["test_episodes"]) != 0:
        raise ValueError("E3 endpoint construction must not access test")
    generator = SyntheticEpisodeGenerator(
        base_seed=int(data["split_seed"]),
        episode_steps=int(data["max_episode_steps"]),
        schedule_policy=policy,
    )
    candidates: list[EndpointCandidate] = []
    for episode_index in range(int(data["validation_episodes"])):
        episode = generator.generate_episode(DatasetSplit.VALIDATION, episode_index)
        for example in generate_queries(episode, lags=data["query_lags"]):
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
    return build_balanced_endpoint_manifest(
        candidates,
        tasks=PRIMARY_TASKS,
        lags=PRIMARY_LAGS,
        examples_per_cell=examples_per_cell,
        dataset_source_sha256=dataset_source_sha256(),
        config_sha256=content_sha256(config),
        generator_policy=policy,
        salt=E3_ENDPOINT_SALT,
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
