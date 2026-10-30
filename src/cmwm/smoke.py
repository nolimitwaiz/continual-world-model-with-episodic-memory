"""Cluster-only engineering smoke suites.

These checks establish executable contracts. Their losses and outputs are not
scientific measurements and must never be used to select a condition.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from cmwm.infrastructure.reproducibility import seed_everything
from cmwm.infrastructure.system_info import collect_system_info
from cmwm.schemas import DatasetSplit
from cmwm.synthetic.audit import audit_g0
from cmwm.synthetic.causal_controls import causal_controls_for_dataset
from cmwm.synthetic.generator import SyntheticEpisodeGenerator
from cmwm.synthetic.manifest import (
    build_dataset_manifest,
    manifest_identity,
    seal_manifest,
    verify_manifest_seal,
)
from cmwm.synthetic.queries import answer_query, generate_queries
from cmwm.synthetic.validation import episode_checksum, validate_dataset


SMOKE_SUITES = ("environment", "generator", "g0-readiness", "components")


def _source_tree_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".yaml", ".toml", ".sbatch", ".md"}
        and ".pytest_cache" not in path.parts
        and "__pycache__" not in path.parts
    )
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _base_result(suite: str, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_type": "cmwm_engineering_smoke",
        "claim_eligible": False,
        "suite": suite,
        "experiment": config["experiment"]["name"],
        "seed": int(config["experiment"]["seed"]),
        "source_tree_sha256": _source_tree_hash(),
        "system": collect_system_info(),
    }


def _environment_smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    import yaml

    seed_everything(int(config["experiment"]["seed"]), deterministic=True)
    left = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
    output = left @ torch.eye(4, device="cuda")
    loss = output.square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("environment smoke produced a non-finite CUDA result")
    result = _base_result("environment", config)
    result.update(
        {
            "passed": True,
            "numpy": np.__version__,
            "pyyaml": yaml.__version__,
            "cuda_tensor_shape": list(output.shape),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
    )
    return result


def _generator_fixture(
    config: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, tuple[Any, ...]]]:
    seed = int(config["data"]["split_seed"])
    generator = SyntheticEpisodeGenerator(base_seed=seed, episode_steps=160)
    episodes = tuple(
        generator.generate_episode(split, index)
        for split in (DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST)
        for index in range(2)
    )
    queries = {episode.episode_id: generate_queries(episode) for episode in episodes}
    return episodes, queries


def _generator_smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    episodes, queries = _generator_fixture(config)
    validate_dataset(episodes)
    for episode in episodes:
        for example in queries[episode.episode_id]:
            if answer_query(episode, example.query) != example.oracle_answer:
                raise RuntimeError(f"oracle answer mismatch: {example.query.query_id}")

    repeat, repeat_queries = _generator_fixture(config)
    checksums = tuple(episode_checksum(episode) for episode in episodes)
    repeat_checksums = tuple(episode_checksum(episode) for episode in repeat)
    if checksums != repeat_checksums:
        raise RuntimeError("generator replay hashes differ")

    manifest = build_dataset_manifest(episodes, queries)
    repeat_manifest = build_dataset_manifest(repeat, repeat_queries)
    seal = seal_manifest(manifest)
    if manifest_identity(manifest) != manifest_identity(repeat_manifest):
        raise RuntimeError("replayed dataset manifest identity differs")
    if not verify_manifest_seal(repeat_manifest, seal):
        raise RuntimeError("manifest seal failed on deterministic replay")

    result = _base_result("generator", config)
    result.update(
        {
            "passed": True,
            "episodes": len(episodes),
            "queries": sum(len(items) for items in queries.values()),
            "manifest_sha256": manifest_identity(manifest),
            "episode_checksums": list(checksums),
        }
    )
    return result


def _g0_fixture(
    config: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, tuple[Any, ...]]]:
    data = config["data"]
    generator = SyntheticEpisodeGenerator(
        base_seed=int(data["split_seed"]), episode_steps=160
    )
    dataset = generator.generate_dataset(
        train_episodes=int(data["train_episodes"]),
        validation_episodes=int(data["validation_episodes"]),
        test_episodes=int(data["test_episodes"]),
    )
    episodes = tuple(
        episode
        for split in (DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST)
        for episode in dataset[split]
    )
    queries = {episode.episode_id: generate_queries(episode) for episode in episodes}
    return episodes, queries


def _tensor_fixtures(device: str) -> tuple[Any, Any, Any]:
    import torch

    from cmwm.models.actions import ActionBatch
    from cmwm.models.queries import QueryBatch
    from cmwm.models.symbolic import SymbolicStateBatch

    state = SymbolicStateBatch(
        agent_position=torch.tensor([[0.0, 0.0], [2.0, 1.0]], device=device),
        held_entity_slot=torch.tensor([-1, 0], device=device),
        entity_positions=torch.tensor(
            [
                [[1.0, 0.0], [3.0, 1.0], [0.0, 0.0]],
                [[2.0, 1.0], [4.0, 2.0], [0.0, 0.0]],
            ],
            device=device,
        ),
        entity_kinds=torch.tensor([[0, 1, 0], [0, 1, 0]], device=device),
        contained_in_slot=torch.tensor([[1, -1, -1], [-1, -1, -1]], device=device),
        entity_is_open=torch.tensor(
            [[False, True, False], [False, False, False]], device=device
        ),
        entity_mask=torch.tensor(
            [[True, True, False], [True, True, False]], device=device
        ),
    )
    action = ActionBatch(
        kind=torch.tensor([0, 7], device=device),
        target_entity_slot=torch.tensor([-1, 0], device=device),
        container_slot=torch.tensor([-1, 1], device=device),
        destination=torch.tensor([[0.0, 0.0], [4.0, 2.0]], device=device),
        destination_mask=torch.tensor([False, True], device=device),
    )
    query = QueryBatch(
        kind=torch.tensor([0, 3], device=device),
        target_entity_slot=torch.tensor([0, 1], device=device),
        target_field=torch.tensor([0, 2], device=device),
        lag=torch.tensor([64, 128], device=device),
        causal_target_kind=torch.tensor([-1, 4], device=device),
        causal_target_entity_slot=torch.tensor([-1, 0], device=device),
        causal_value_token=torch.tensor([-1, 3], device=device),
    )
    return state, action, query


def _build_modules(device: str) -> Any:
    from torch import nn

    from cmwm.models.memory_writer import SharedKeyValueMemoryWriter
    from cmwm.models.queries import SharedQueryHead, StructuredQueryEncoder
    from cmwm.models.symbolic import CanonicalSymbolicStepAdapter
    from cmwm.models.temporal import FixedGRUTemporalCore

    return nn.ModuleDict(
        {
            "step": CanonicalSymbolicStepAdapter(
                max_entities=8, grid_width=6, grid_height=4
            ),
            "temporal": FixedGRUTemporalCore(),
            "writer": SharedKeyValueMemoryWriter(),
            "query": StructuredQueryEncoder(
                max_entities=8,
                field_vocab_size=6,
                causal_target_vocab_size=12,
                answer_vocab_size=32,
            ),
            "head": SharedQueryHead(answer_vocab_size=32),
        }
    ).to(device)


def _segments_for_condition(condition: str, length: int) -> tuple[Any, ...]:
    from cmwm.models.boundaries import (
        FixedWindowSegmenter,
        OracleSegmenter,
        RandomMatchedSegmenter,
        segments_from_boundaries,
    )

    oracle_lengths = (2, 3, 3)
    if sum(oracle_lengths) != length:
        raise ValueError("component fixture length drifted")
    if condition == "frame_memory":
        boundaries = tuple(range(1, length))
    elif condition == "fixed_chunks":
        boundaries = FixedWindowSegmenter(width=4).boundaries(length=length)
    elif condition == "random_matched":
        boundaries = RandomMatchedSegmenter(
            reference_segment_lengths=oracle_lengths, seed=17
        ).boundaries(length=length)
    elif condition == "oracle_events":
        boundaries = OracleSegmenter((2, 5)).boundaries(length=length)
    else:
        raise ValueError(f"unknown smoke condition: {condition}")
    return segments_from_boundaries(length, boundaries)


def _forward_losses(modules: Any) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch.nn import functional as F

    from cmwm.models.queries import RetrievedMemoryBatch

    state, action, query = _tensor_fixtures("cuda")
    encoded = modules["step"](state, action)
    sequence = encoded.unsqueeze(1).expand(-1, 8, -1).contiguous()
    temporal = modules["temporal"](sequence).sequence
    query_encoding = modules["query"](query)
    current = temporal[:, -1]
    answer_targets = torch.tensor([1, 2], device="cuda")
    losses = []
    conditions: dict[str, Any] = {}

    empty = RetrievedMemoryBatch.empty(
        batch_size=2, device="cuda", dtype=current.dtype
    )
    context_output = modules["head"](current, query_encoding, empty)
    losses.append(F.cross_entropy(context_output.answer_logits, answer_targets))
    conditions["context_only"] = {
        "answer_shape": list(context_output.answer_logits.shape),
        "retrieved_records": 0,
    }

    for condition in ("frame_memory", "fixed_chunks", "random_matched", "oracle_events"):
        segments = _segments_for_condition(condition, temporal.shape[1])
        keys = []
        values = []
        for segment in segments[:4]:
            states = temporal[:, segment.start : segment.end]
            mask = torch.ones(states.shape[:2], device="cuda", dtype=torch.bool)
            written = modules["writer"](states, mask)
            keys.append(written.keys)
            values.append(written.values)
        key_tensor = torch.stack(keys, dim=1)
        value_tensor = torch.stack(values, dim=1)
        retrieved = RetrievedMemoryBatch(
            key_tensor,
            value_tensor,
            torch.ones(key_tensor.shape[:2], device="cuda", dtype=torch.bool),
        )
        output = modules["head"](current, query_encoding, retrieved)
        losses.append(F.cross_entropy(output.answer_logits, answer_targets))
        candidate_targets = torch.zeros(2, device="cuda", dtype=torch.long)
        losses.append(F.cross_entropy(output.candidate_logits, candidate_targets))
        conditions[condition] = {
            "segment_lengths": [segment.length for segment in segments],
            "records_written": len(segments),
            "records_retrieved": len(keys),
            "answer_shape": list(output.answer_logits.shape),
            "candidate_shape": list(output.candidate_logits.shape),
        }

    if sorted(conditions["random_matched"]["segment_lengths"]) != [2, 3, 3]:
        raise RuntimeError("random-matched smoke changed the length distribution")
    return torch.stack(losses).sum(), conditions


def _components_smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    from cmwm.models.memory_writer import MemoryRecordLayout

    seed_everything(int(config["experiment"]["seed"]), deterministic=True)
    modules = _build_modules("cuda")
    optimizer = torch.optim.AdamW(modules.parameters(), lr=1.0e-4)
    optimizer.zero_grad(set_to_none=True)
    loss, conditions = _forward_losses(modules)
    if not torch.isfinite(loss):
        raise RuntimeError("component smoke produced a non-finite loss")
    loss.backward()
    gradients = [
        parameter.grad for parameter in modules.parameters() if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("component smoke produced missing or non-finite gradients")
    optimizer.step()

    run_root_value = os.environ.get("CMWM_RUN_ROOT")
    if not run_root_value:
        raise RuntimeError("CMWM_RUN_ROOT is required for checkpoint smoke")
    run_root = Path(run_root_value).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cmwm-smoke-", dir=run_root) as temporary:
        checkpoint = Path(temporary) / "roundtrip.pt"
        torch.save(
            {"model": modules.state_dict(), "optimizer": optimizer.state_dict()},
            checkpoint,
        )
        loaded = torch.load(checkpoint, map_location="cuda", weights_only=False)
        restored = _build_modules("cuda")
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.0e-4)
        restored.load_state_dict(loaded["model"])
        restored_optimizer.load_state_dict(loaded["optimizer"])
        for name, value in modules.state_dict().items():
            if not torch.equal(value, restored.state_dict()[name]):
                raise RuntimeError(f"checkpoint roundtrip mismatch: {name}")

    layout = MemoryRecordLayout()
    result = _base_result("components", config)
    result.update(
        {
            "passed": True,
            "engineering_loss_finite": True,
            "trainable_parameters": sum(
                parameter.numel() for parameter in modules.parameters()
            ),
            "gradient_tensors": len(gradients),
            "memory_record_bytes": layout.record_bytes,
            "memory_capacity_bytes": layout.capacity_bytes(),
            "checkpoint_roundtrip": True,
            "conditions": conditions,
        }
    )
    return result


def _g0_readiness_smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    episodes, queries = _g0_fixture(config)
    controls = causal_controls_for_dataset(episodes, queries)
    from cmwm.g0 import _segment_evidence

    oracle_lengths, random_lengths = _segment_evidence(
        episodes, seed=int(config["data"]["split_seed"]) + 91_771
    )
    report = audit_g0(
        episodes,
        queries,
        causal_candidates_by_query=controls,
        oracle_segment_lengths_by_episode=oracle_lengths,
        random_segment_lengths_by_episode=random_lengths,
    )
    result = _base_result("g0-readiness", config)
    result.update(
        {
            "passed": report.passed,
            "episodes": report.episode_count,
            "queries": report.query_count,
            "error_codes": sorted({finding.code for finding in report.errors}),
            "warning_codes": sorted(
                {
                    finding.code
                    for finding in report.findings
                    if finding.severity.value == "warning"
                }
            ),
            "note": "A false result is a research blocker, not a failed hypothesis.",
        }
    )
    return result


def run_smoke(suite: str, config: Mapping[str, Any]) -> dict[str, Any]:
    if suite not in SMOKE_SUITES:
        raise ValueError(f"unknown smoke suite: {suite}")
    runners = {
        "environment": _environment_smoke,
        "generator": _generator_smoke,
        "g0-readiness": _g0_readiness_smoke,
        "components": _components_smoke,
    }
    result = runners[suite](config)
    json.dumps(result, sort_keys=True)
    return result
