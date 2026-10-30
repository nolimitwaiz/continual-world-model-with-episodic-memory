"""Cluster-only construction of a sealed G0 dataset attestation."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from random import Random
from typing import Any, Mapping

from cmwm.infrastructure.artifacts import atomic_write_json
from cmwm.models.boundaries import RandomMatchedSegmenter, segments_from_boundaries
from cmwm.schemas import DatasetSplit
from cmwm.synthetic.audit import audit_g0
from cmwm.synthetic.causal_controls import causal_controls_for_dataset
from cmwm.synthetic.generator import SyntheticEpisodeGenerator
from cmwm.synthetic.manifest import content_sha256
from cmwm.synthetic.queries import generate_queries


def dataset_source_sha256() -> str:
    """Fingerprint every source file that can change dataset eligibility."""

    package_root = Path(__file__).resolve().parent
    relative_paths = (
        "g0.py",
        "schemas.py",
        "models/boundaries.py",
        "synthetic/audit.py",
        "synthetic/causal_controls.py",
        "synthetic/generator.py",
        "synthetic/manifest.py",
        "synthetic/queries.py",
        "synthetic/validation.py",
        "synthetic/world.py",
    )
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = package_root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _segment_evidence(
    episodes: tuple[Any, ...], *, seed: int
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    oracle: dict[str, tuple[int, ...]] = {}
    random_matched: dict[str, tuple[int, ...]] = {}
    rng = Random(seed)
    for episode in episodes:
        length = len(episode.steps)
        true_boundaries = tuple(
            step for step in episode.boundary_steps if 0 < step < length
        )
        oracle_segments = segments_from_boundaries(length, true_boundaries)
        oracle_lengths = tuple(segment.length for segment in oracle_segments)
        randomized_boundaries = RandomMatchedSegmenter(
            oracle_lengths, seed=rng.randrange(2**31)
        ).boundaries(length=length)
        randomized_segments = segments_from_boundaries(length, randomized_boundaries)
        oracle[episode.episode_id] = oracle_lengths
        random_matched[episode.episode_id] = tuple(
            segment.length for segment in randomized_segments
        )
    return oracle, random_matched


def build_g0_artifact(
    config: Mapping[str, Any], output: str | Path
) -> dict[str, Any]:
    """Generate, audit, and seal the declared dataset or emit a failed report.

    A failed report is useful engineering evidence but cannot unlock training.
    The present scaffold is expected to fail until causal candidate controls and
    the remaining generator-diversity requirements are implemented.
    """

    data = config["data"]
    generator = SyntheticEpisodeGenerator(
        base_seed=int(data["split_seed"]),
        episode_steps=int(data["max_episode_steps"]),
        schedule_policy=str(data.get("generator_policy", "random_v1")),
    )
    generated = generator.generate_dataset(
        train_episodes=int(data["train_episodes"]),
        validation_episodes=int(data["validation_episodes"]),
        test_episodes=int(data["test_episodes"]),
    )
    episodes = tuple(
        episode
        for split in (DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST)
        for episode in generated[split]
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
    report = audit_g0(
        episodes,
        queries,
        causal_candidates_by_query=causal_controls,
        oracle_segment_lengths_by_episode=oracle_lengths,
        random_segment_lengths_by_episode=random_lengths,
    )
    dataset_contract = {
        "generator_policy": str(data.get("generator_policy", "random_v1")),
        "split_seed": int(data["split_seed"]),
        "train_episodes": int(data["train_episodes"]),
        "validation_episodes": int(data["validation_episodes"]),
        "test_episodes": int(data["test_episodes"]),
    }
    artifact: dict[str, Any] = {
        "artifact_type": (
            "cmwm_g0_attestation" if report.passed else "cmwm_g0_failed_report"
        ),
        "passed": report.passed,
        "dataset_contract": dataset_contract,
        "dataset_source_sha256": dataset_source_sha256(),
        "episode_count": report.episode_count,
        "query_count": report.query_count,
        "errors": [
            {
                "code": finding.code,
                "message": finding.message,
                "episode_ids": list(finding.episode_ids),
                "query_ids": list(finding.query_ids),
            }
            for finding in report.errors
        ],
        "findings": [
            {
                "severity": finding.severity.value,
                "code": finding.code,
                "message": finding.message,
            }
            for finding in report.findings
        ],
        "balance": asdict(report.balance),
        "storage_pressure": asdict(report.storage_pressure),
        "segment_length_match": asdict(report.segment_length_match),
        "manifest_sha256": (
            report.manifest_seal.manifest_sha256
            if report.manifest_seal is not None
            else None
        ),
        "seal_sha256": (
            content_sha256(report.manifest_seal)
            if report.manifest_seal is not None
            else None
        ),
        "manifest_seal": (
            asdict(report.manifest_seal) if report.manifest_seal is not None else None
        ),
        "note": (
            "This attestation unlocks only the matching declared configuration."
            if report.passed
            else "Training remains locked until every G0 error is resolved."
        ),
    }
    atomic_write_json(output, artifact)
    return artifact
