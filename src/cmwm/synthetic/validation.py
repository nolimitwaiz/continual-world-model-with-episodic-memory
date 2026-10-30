"""Leakage- and consistency-oriented validation for generated episodes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from enum import Enum
from typing import Any, Iterable

from cmwm.schemas import DatasetSplit, Episode, Mechanism


class ValidationError(ValueError):
    pass


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def episode_checksum(episode: Episode) -> str:
    payload = json.dumps(
        _json_value(asdict(episode)), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_episode(episode: Episode) -> None:
    """Raise when transition, boundary, intervention, or split metadata is invalid."""

    expected_before = episode.initial_state
    seen_event_ids: set[str] = set()
    for expected_step, record in enumerate(episode.steps, start=1):
        if record.step_index != expected_step:
            raise ValidationError("step indices must be contiguous and one-based")
        if record.before != expected_before:
            raise ValidationError("step before-state does not match previous after-state")
        if record.after.step_index != expected_step:
            raise ValidationError("after-state step index mismatch")
        if record.counterfactual_after.step_index != expected_step:
            raise ValidationError("counterfactual step index mismatch")
        event_ids = {event.event_id for event in record.events}
        if seen_event_ids & event_ids:
            raise ValidationError("event IDs must be unique within an episode")
        seen_event_ids.update(event_ids)
        if not set(record.boundary.event_ids).issubset(event_ids):
            raise ValidationError("boundary references an unknown event")
        if record.boundary.is_boundary != bool(record.boundary.reasons):
            raise ValidationError("boundary flag and reasons disagree")
        for edge in record.causal_edges:
            if edge.evidence.factual_value == edge.evidence.counterfactual_value:
                raise ValidationError("causal edge lacks intervention contrast")
        expected_before = record.after

    metadata = episode.split_metadata
    if metadata.dataset_split is not DatasetSplit.TEST and metadata.held_out_family:
        raise ValidationError("non-test episode declares a held-out family")
    if metadata.held_out_family not in {None, "HF-A", "HF-B"}:
        raise ValidationError("unknown held-out family")
    if metadata.held_out_family == "HF-B" and Mechanism.CONTAINER_TRANSPORT not in (
        metadata.episode_mechanisms
    ):
        raise ValidationError("HF-B must contain the held-out transport mechanism")
    if metadata.dataset_split is not DatasetSplit.TEST and (
        Mechanism.CONTAINER_TRANSPORT in metadata.episode_mechanisms
    ):
        raise ValidationError("held-out transport leaked into train/validation")


def validate_dataset(episodes: Iterable[Episode]) -> None:
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    test_families: set[str] = set()
    count = 0
    for episode in episodes:
        count += 1
        validate_episode(episode)
        if episode.episode_id in seen_ids:
            raise ValidationError("duplicate episode ID")
        if episode.split_metadata.episode_seed in seen_seeds:
            raise ValidationError("episode seed reused across splits")
        seen_ids.add(episode.episode_id)
        seen_seeds.add(episode.split_metadata.episode_seed)
        if episode.split_metadata.held_out_family:
            test_families.add(episode.split_metadata.held_out_family)
    if count == 0:
        raise ValidationError("dataset is empty")
    if test_families and test_families != {"HF-A", "HF-B"}:
        raise ValidationError("test data must contain both HF-A and HF-B")
