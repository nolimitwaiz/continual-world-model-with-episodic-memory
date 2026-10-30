#!/usr/bin/env python3
"""Audit the shortcut-resistant E3 train/validation construction without test."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from cmwm.config import load_config
from cmwm.g0 import _segment_evidence, dataset_source_sha256
from cmwm.infrastructure.artifacts import atomic_write_json
from cmwm.schemas import DatasetSplit
from cmwm.synthetic.audit import G0AuditPolicy, audit_g0
from cmwm.synthetic.causal_controls import causal_controls_for_dataset
from cmwm.synthetic.generator import SHORTCUT_RESISTANT_POLICY, SyntheticEpisodeGenerator
from cmwm.synthetic.manifest import content_sha256
from cmwm.synthetic.queries import generate_queries


def audit_development(
    config_path: Path,
    *,
    train_episodes: int | None = None,
    validation_episodes: int | None = None,
) -> dict[str, object]:
    config = load_config(config_path)
    data = config["data"]
    policy_name = str(data.get("generator_policy", "random_v1"))
    if policy_name != SHORTCUT_RESISTANT_POLICY:
        raise ValueError(f"E3 requires {SHORTCUT_RESISTANT_POLICY}")
    if int(data["test_episodes"]) != 0:
        raise ValueError("E3 construction must not access test")
    train_count = int(data["train_episodes"]) if train_episodes is None else train_episodes
    validation_count = (
        int(data["validation_episodes"])
        if validation_episodes is None
        else validation_episodes
    )
    if train_count <= 0 or validation_count <= 0:
        raise ValueError("development episode counts must be positive")

    generator = SyntheticEpisodeGenerator(
        base_seed=int(data["split_seed"]),
        episode_steps=int(data["max_episode_steps"]),
        schedule_policy=policy_name,
    )
    episodes = tuple(
        generator.generate_episode(split, index)
        for split, count in (
            (DatasetSplit.TRAIN, train_count),
            (DatasetSplit.VALIDATION, validation_count),
        )
        for index in range(count)
    )
    queries = {
        episode.episode_id: generate_queries(
            episode, lags=tuple(int(value) for value in data["query_lags"])
        )
        for episode in episodes
    }
    oracle_lengths, random_lengths = _segment_evidence(
        episodes, seed=int(data["split_seed"]) + 91_771
    )
    causal_controls = causal_controls_for_dataset(episodes, queries)
    audit_policy = G0AuditPolicy(
        required_test_families=(),
        max_dominant_label_fraction=0.60,
        shortcut_accuracy_threshold=0.80,
        shortcut_uplift_threshold=0.05,
        model_max_entities=int(config["prototype1"]["state_adapter"]["max_entities"]),
    )
    report = audit_g0(
        episodes,
        queries,
        policy=audit_policy,
        causal_candidates_by_query=causal_controls,
        oracle_segment_lengths_by_episode=oracle_lengths,
        random_segment_lengths_by_episode=random_lengths,
    )
    full_contract = (
        train_count == int(data["train_episodes"])
        and validation_count == int(data["validation_episodes"])
    )
    passing_attestation = report.passed and full_contract
    seal = asdict(report.manifest_seal) if report.manifest_seal is not None else None
    seal_sha256 = (
        content_sha256(report.manifest_seal)
        if report.manifest_seal is not None
        else None
    )
    return {
        "artifact_type": (
            "cmwm_g0_attestation"
            if passing_attestation
            else "cmwm_e3_development_g0_readiness_v1"
        ),
        "claim_eligible": False,
        "scope": "train_and_validation_only",
        "test_accessed": False,
        "passed": report.passed,
        "full_declared_development": full_contract,
        "generator_policy": policy_name,
        "dataset_source_sha256": dataset_source_sha256(),
        "config_sha256": content_sha256(config),
        "dataset_contract": {
            "generator_policy": policy_name,
            "split_seed": int(data["split_seed"]),
            "train_episodes": train_count,
            "validation_episodes": validation_count,
            "test_episodes": 0,
        },
        "episode_count": report.episode_count,
        "query_count": report.query_count,
        "errors": [asdict(finding) for finding in report.errors],
        "findings": [asdict(finding) for finding in report.findings],
        "balance": asdict(report.balance),
        "storage_pressure": asdict(report.storage_pressure),
        "segment_length_match": asdict(report.segment_length_match),
        "label_diagnostics": [asdict(item) for item in report.label_diagnostics],
        "shortcut_diagnostics": [
            asdict(item) for item in report.shortcut_diagnostics
        ],
        "manifest_sha256": (
            report.manifest_seal.manifest_sha256
            if report.manifest_seal is not None
            else None
        ),
        "manifest_seal_sha256": seal_sha256,
        "seal_sha256": seal_sha256,
        "manifest_seal": seal,
        "note": (
            "E3 construction passed stricter model-visible anti-shortcut gates; "
            "this authorizes endpoint freezing, not GPU training."
            if passing_attestation
            else "This readiness artifact cannot authorize training."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int)
    parser.add_argument("--validation-episodes", type=int)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite readiness artifact: {args.output}")
    report = audit_development(
        args.config,
        train_episodes=args.train_episodes,
        validation_episodes=args.validation_episodes,
    )
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["passed"] or not args.require_pass else 4


if __name__ == "__main__":
    raise SystemExit(main())
