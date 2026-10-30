"""Tensorization for the controlled symbolic CMWM-E1 experiment.

The tensorizer is deliberately condition-blind.  A single episode/query batch
is shared by context, frame, fixed, random-matched, oracle, and learned event
conditions.  Oracle event labels are carried as evaluator/control signals and
are consumed only when the selected segmentation policy explicitly permits it.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from random import Random
from typing import Sequence

import torch
from torch import Tensor

from cmwm.models.actions import ACTION_KIND_COUNT, ActionBatch
from cmwm.models.queries import QUERY_KIND_COUNT, QueryBatch
from cmwm.models.symbolic import ENTITY_KIND_COUNT, SymbolicStateBatch
from cmwm.schemas import (
    ActionKind,
    EntityKind,
    Episode,
    QueryExample,
    QueryKind,
    WorldState,
)


TARGET_FIELDS = ("position", "contained_in", "is_open")
CAUSAL_TARGET_KINDS = (
    "agent.position",
    "agent.held_object_id",
    "entity.position",
    "entity.contained_in",
    "entity.is_open",
)

_ACTION_INDEX = {kind: index for index, kind in enumerate(ActionKind)}
_ENTITY_KIND_INDEX = {kind: index for index, kind in enumerate(EntityKind)}
_QUERY_KIND_INDEX = {kind: index for index, kind in enumerate(QueryKind)}
_FIELD_INDEX = {field: index for index, field in enumerate(TARGET_FIELDS)}
_CAUSAL_TARGET_INDEX = {
    target: index for index, target in enumerate(CAUSAL_TARGET_KINDS)
}

if len(_ACTION_INDEX) != ACTION_KIND_COUNT:
    raise RuntimeError("ActionKind and ACTION_KIND_COUNT disagree")
if len(_ENTITY_KIND_INDEX) != ENTITY_KIND_COUNT:
    raise RuntimeError("EntityKind and ENTITY_KIND_COUNT disagree")
if len(_QUERY_KIND_INDEX) != QUERY_KIND_COUNT:
    raise RuntimeError("QueryKind and QUERY_KIND_COUNT disagree")


@dataclass(frozen=True)
class AnswerVocabulary:
    """Disjoint fixed answer ranges for positions, containment, and booleans."""

    grid_width: int
    grid_height: int
    max_entities: int

    def __post_init__(self) -> None:
        if self.grid_width <= 1 or self.grid_height <= 1:
            raise ValueError("grid dimensions must exceed one")
        if self.max_entities <= 0:
            raise ValueError("max_entities must be positive")

    @property
    def position_count(self) -> int:
        return self.grid_width * self.grid_height

    @property
    def containment_offset(self) -> int:
        return self.position_count

    @property
    def boolean_offset(self) -> int:
        return self.containment_offset + self.max_entities + 1

    @property
    def size(self) -> int:
        return self.boolean_offset + 2

    def encode(
        self,
        *,
        field: str,
        value: object,
        entity_slots: dict[str, int],
    ) -> int:
        if field == "position":
            if not (
                isinstance(value, tuple)
                and len(value) == 2
                and all(isinstance(item, int) for item in value)
            ):
                raise ValueError("position answer must be an integer pair")
            x, y = value
            if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
                raise ValueError("position answer lies outside the grid")
            return y * self.grid_width + x
        if field == "contained_in":
            if value is None:
                relation_index = 0
            elif isinstance(value, str) and value in entity_slots:
                relation_index = entity_slots[value] + 1
            else:
                raise ValueError("containment answer references an unknown entity")
            return self.containment_offset + relation_index
        if field == "is_open":
            if not isinstance(value, bool):
                raise ValueError("is_open answer must be boolean")
            return self.boolean_offset + int(value)
        raise ValueError(f"unsupported answer field: {field}")

    def encode_causal_value(
        self,
        *,
        target: str,
        value: object,
        entity_slots: dict[str, int],
    ) -> int:
        if target.endswith(".position"):
            field = "position"
        elif target.endswith(".contained_in") or target == "agent.held_object_id":
            field = "contained_in"
        elif target.endswith(".is_open"):
            field = "is_open"
        else:
            raise ValueError(f"unsupported causal value target: {target}")
        return self.encode(field=field, value=value, entity_slots=entity_slots)


@dataclass(frozen=True)
class SymbolicSequenceBatch:
    """Padded episode tensors; all sequence tensors begin with [batch, steps]."""

    agent_position: Tensor
    held_entity_slot: Tensor
    entity_positions: Tensor
    entity_kinds: Tensor
    contained_in_slot: Tensor
    entity_is_open: Tensor
    entity_mask: Tensor
    action_kind: Tensor
    action_target_entity_slot: Tensor
    action_container_slot: Tensor
    action_destination: Tensor
    action_destination_mask: Tensor
    step_mask: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.step_mask.shape[0])

    @property
    def step_count(self) -> int:
        return int(self.step_mask.shape[1])

    @property
    def entity_count(self) -> int:
        return int(self.entity_positions.shape[2])

    def validate(self) -> None:
        batch, steps = self.step_mask.shape
        entities = self.entity_count
        expected_bs = (batch, steps)
        if self.step_mask.dtype is not torch.bool:
            raise TypeError("step_mask must be boolean")
        if tuple(self.agent_position.shape) != (batch, steps, 2):
            raise ValueError("agent_position must have shape [batch, steps, 2]")
        if tuple(self.entity_positions.shape) != (batch, steps, entities, 2):
            raise ValueError(
                "entity_positions must have shape [batch, steps, entities, 2]"
            )
        for name, tensor in (
            ("held_entity_slot", self.held_entity_slot),
            ("action_kind", self.action_kind),
            ("action_target_entity_slot", self.action_target_entity_slot),
            ("action_container_slot", self.action_container_slot),
            ("action_destination_mask", self.action_destination_mask),
        ):
            if tuple(tensor.shape) != expected_bs:
                raise ValueError(f"{name} must have shape [batch, steps]")
        for name, tensor in (
            ("entity_kinds", self.entity_kinds),
            ("contained_in_slot", self.contained_in_slot),
            ("entity_is_open", self.entity_is_open),
            ("entity_mask", self.entity_mask),
        ):
            if tuple(tensor.shape) != (batch, steps, entities):
                raise ValueError(f"{name} must have shape [batch, steps, entities]")
        if tuple(self.action_destination.shape) != (batch, steps, 2):
            raise ValueError("action_destination must have shape [batch, steps, 2]")
        if self.entity_mask.dtype is not torch.bool:
            raise TypeError("entity_mask must be boolean")
        if self.action_destination_mask.dtype is not torch.bool:
            raise TypeError("action_destination_mask must be boolean")

    def flattened_state(self) -> SymbolicStateBatch:
        self.validate()
        batch_steps = self.batch_size * self.step_count
        return SymbolicStateBatch(
            agent_position=self.agent_position.reshape(batch_steps, 2),
            held_entity_slot=self.held_entity_slot.reshape(batch_steps),
            entity_positions=self.entity_positions.reshape(
                batch_steps, self.entity_count, 2
            ),
            entity_kinds=self.entity_kinds.reshape(batch_steps, self.entity_count),
            contained_in_slot=self.contained_in_slot.reshape(
                batch_steps, self.entity_count
            ),
            entity_is_open=self.entity_is_open.reshape(
                batch_steps, self.entity_count
            ),
            entity_mask=self.entity_mask.reshape(batch_steps, self.entity_count),
        )

    def flattened_action(self) -> ActionBatch:
        self.validate()
        batch_steps = self.batch_size * self.step_count
        return ActionBatch(
            kind=self.action_kind.reshape(batch_steps),
            target_entity_slot=self.action_target_entity_slot.reshape(batch_steps),
            container_slot=self.action_container_slot.reshape(batch_steps),
            destination=self.action_destination.reshape(batch_steps, 2),
            destination_mask=self.action_destination_mask.reshape(batch_steps),
        )


@dataclass(frozen=True)
class CMWMBatch:
    sequence: SymbolicSequenceBatch
    query: QueryBatch
    ask_steps: Tensor
    reference_steps: Tensor
    oracle_boundaries: Tensor
    answer_targets: Tensor
    causal_source_steps: Tensor
    query_ids: tuple[str, ...]
    episode_ids: tuple[str, ...]
    answer_vocab_size: int

    @property
    def batch_size(self) -> int:
        return self.sequence.batch_size

    def validate(self) -> None:
        self.sequence.validate()
        batch = self.batch_size
        self.query.validate(
            max_entities=self.sequence.entity_count,
            field_vocab_size=len(TARGET_FIELDS),
            causal_target_vocab_size=len(CAUSAL_TARGET_KINDS),
            answer_vocab_size=self.answer_vocab_size,
        )
        for name, tensor in (
            ("ask_steps", self.ask_steps),
            ("reference_steps", self.reference_steps),
            ("answer_targets", self.answer_targets),
            ("causal_source_steps", self.causal_source_steps),
        ):
            if tuple(tensor.shape) != (batch,):
                raise ValueError(f"{name} must have shape [batch]")
        if tuple(self.oracle_boundaries.shape) != tuple(self.sequence.step_mask.shape):
            raise ValueError("oracle_boundaries must have shape [batch, steps]")
        if self.oracle_boundaries.dtype is not torch.bool:
            raise TypeError("oracle_boundaries must be boolean")
        if len(self.query_ids) != batch or len(self.episode_ids) != batch:
            raise ValueError("identifier collections must match the batch size")
        lengths = self.sequence.step_mask.sum(dim=1)
        if torch.any(self.ask_steps <= 0) or torch.any(self.ask_steps > lengths):
            raise ValueError("ask_steps must lie inside each episode")
        if torch.any(self.reference_steps < 0) or torch.any(
            self.reference_steps > self.ask_steps
        ):
            raise ValueError("reference_steps must not follow ask_steps")


def _episode_slot_map(episode: Episode, *, max_entities: int) -> dict[str, int]:
    entity_ids = sorted(entity.entity_id for entity in episode.initial_state.entities)
    if len(entity_ids) > max_entities:
        raise ValueError("episode exceeds max_entities")
    seed_material = (
        f"cmwm-slot-v1:{episode.split_metadata.episode_seed}".encode("utf-8")
    )
    seed = int.from_bytes(sha256(seed_material).digest()[:8], "big")
    slots = list(range(max_entities))
    Random(seed).shuffle(slots)
    # Only the occupied prefix participates in the strict zip.  The remaining
    # slots are padding capacity, not missing entity IDs.
    return dict(zip(entity_ids, slots[: len(entity_ids)], strict=True))


def _write_state(
    *,
    state: WorldState,
    batch_index: int,
    step_index: int,
    slots: dict[str, int],
    agent_position: Tensor,
    held_entity_slot: Tensor,
    entity_positions: Tensor,
    entity_kinds: Tensor,
    contained_in_slot: Tensor,
    entity_is_open: Tensor,
    entity_mask: Tensor,
) -> None:
    agent_position[batch_index, step_index] = torch.tensor(
        state.agent_position.as_tuple(), device=agent_position.device
    )
    held_entity_slot[batch_index, step_index] = (
        -1 if state.held_object_id is None else slots[state.held_object_id]
    )
    for entity in state.entities:
        slot = slots[entity.entity_id]
        entity_positions[batch_index, step_index, slot] = torch.tensor(
            entity.position.as_tuple(), device=entity_positions.device
        )
        entity_kinds[batch_index, step_index, slot] = _ENTITY_KIND_INDEX[entity.kind]
        contained_in_slot[batch_index, step_index, slot] = (
            -1 if entity.contained_in is None else slots[entity.contained_in]
        )
        entity_is_open[batch_index, step_index, slot] = entity.is_open
        entity_mask[batch_index, step_index, slot] = True


def _causal_target_parts(
    target: str, slots: dict[str, int]
) -> tuple[int, int]:
    if target.startswith("agent."):
        return _CAUSAL_TARGET_INDEX[target], -1
    pieces = target.split(".")
    if len(pieces) != 3 or pieces[0] != "entity":
        raise ValueError(f"unsupported causal target: {target}")
    _, entity_id, field = pieces
    return _CAUSAL_TARGET_INDEX[f"entity.{field}"], slots[entity_id]


def _causal_source_step(episode: Episode, answer: object) -> int:
    if not isinstance(answer, str):
        raise ValueError("causal answer must be an action source identifier")
    matching_steps = [
        record.step_index
        for record in episode.steps
        if any(edge.source == answer for edge in record.causal_edges)
    ]
    if len(matching_steps) != 1:
        raise ValueError("causal source must occur in exactly one episode step")
    return matching_steps[0]


def tensorize_examples(
    examples: Sequence[tuple[Episode, QueryExample]],
    *,
    max_entities: int,
    device: torch.device | str = "cpu",
) -> tuple[CMWMBatch, AnswerVocabulary]:
    """Convert episode/query pairs into a padded, condition-independent batch."""

    if not examples:
        raise ValueError("examples cannot be empty")
    width = examples[0][0].config.width
    height = examples[0][0].config.height
    if any(
        episode.config.width != width or episode.config.height != height
        for episode, _ in examples
    ):
        raise ValueError("all examples in a batch must use one grid shape")
    answer_vocabulary = AnswerVocabulary(width, height, max_entities)
    batch = len(examples)
    steps = max(len(episode.steps) for episode, _ in examples)
    entities = max_entities

    def zeros_float(*shape: int) -> Tensor:
        return torch.zeros(*shape, device=device, dtype=torch.float32)

    def zeros_long(*shape: int) -> Tensor:
        return torch.zeros(*shape, device=device, dtype=torch.long)

    agent_position = zeros_float(batch, steps, 2)
    held_entity_slot = torch.full((batch, steps), -1, device=device, dtype=torch.long)
    entity_positions = zeros_float(batch, steps, entities, 2)
    entity_kinds = zeros_long(batch, steps, entities)
    contained_in_slot = torch.full(
        (batch, steps, entities), -1, device=device, dtype=torch.long
    )
    entity_is_open = torch.zeros(
        batch, steps, entities, device=device, dtype=torch.bool
    )
    entity_mask = torch.zeros(batch, steps, entities, device=device, dtype=torch.bool)
    action_kind = zeros_long(batch, steps)
    action_target = torch.full((batch, steps), -1, device=device, dtype=torch.long)
    action_container = torch.full((batch, steps), -1, device=device, dtype=torch.long)
    action_destination = zeros_float(batch, steps, 2)
    action_destination_mask = torch.zeros(batch, steps, device=device, dtype=torch.bool)
    step_mask = torch.zeros(batch, steps, device=device, dtype=torch.bool)
    oracle_boundaries = torch.zeros(batch, steps, device=device, dtype=torch.bool)

    query_kind: list[int] = []
    query_entity: list[int] = []
    query_field: list[int] = []
    query_lag: list[int] = []
    causal_kind: list[int] = []
    causal_entity: list[int] = []
    causal_value: list[int] = []
    ask_steps: list[int] = []
    reference_steps: list[int] = []
    answer_targets: list[int] = []
    causal_source_steps: list[int] = []
    query_ids: list[str] = []
    episode_ids: list[str] = []

    for batch_index, (episode, example) in enumerate(examples):
        slots = _episode_slot_map(episode, max_entities=max_entities)
        episode_ids.append(episode.episode_id)
        query_ids.append(example.query.query_id)
        for time_index, record in enumerate(episode.steps):
            _write_state(
                state=record.after,
                batch_index=batch_index,
                step_index=time_index,
                slots=slots,
                agent_position=agent_position,
                held_entity_slot=held_entity_slot,
                entity_positions=entity_positions,
                entity_kinds=entity_kinds,
                contained_in_slot=contained_in_slot,
                entity_is_open=entity_is_open,
                entity_mask=entity_mask,
            )
            action = record.action
            action_kind[batch_index, time_index] = _ACTION_INDEX[action.kind]
            action_target[batch_index, time_index] = (
                -1 if action.target_id is None else slots[action.target_id]
            )
            action_container[batch_index, time_index] = (
                -1 if action.container_id is None else slots[action.container_id]
            )
            if action.destination is not None:
                action_destination[batch_index, time_index] = torch.tensor(
                    action.destination.as_tuple(), device=device
                )
                action_destination_mask[batch_index, time_index] = True
            step_mask[batch_index, time_index] = True
            if record.boundary.is_boundary and record.step_index < len(episode.steps):
                oracle_boundaries[batch_index, record.step_index] = True

        query = example.query
        query_kind.append(_QUERY_KIND_INDEX[query.kind])
        query_lag.append(query.ask_step - query.reference_step)
        ask_steps.append(query.ask_step)
        reference_steps.append(query.reference_step)
        if query.kind is QueryKind.CAUSAL_PREDECESSOR:
            if query.causal_target is None:
                raise ValueError("causal query is missing its target")
            causal_target_kind, causal_target_entity = _causal_target_parts(
                query.causal_target, slots
            )
            query_entity.append(-1)
            query_field.append(-1)
            causal_kind.append(causal_target_kind)
            causal_entity.append(causal_target_entity)
            causal_value.append(
                answer_vocabulary.encode_causal_value(
                    target=query.causal_target,
                    value=query.causal_value,
                    entity_slots=slots,
                )
            )
            answer_targets.append(-100)
            causal_source_steps.append(
                _causal_source_step(episode, example.oracle_answer)
            )
        else:
            if query.target_entity_id is None or query.target_field is None:
                raise ValueError("state query is missing its target")
            query_entity.append(slots[query.target_entity_id])
            query_field.append(_FIELD_INDEX[query.target_field])
            causal_kind.append(-1)
            causal_entity.append(-1)
            causal_value.append(-1)
            answer_targets.append(
                answer_vocabulary.encode(
                    field=query.target_field,
                    value=example.oracle_answer,
                    entity_slots=slots,
                )
            )
            causal_source_steps.append(-1)

    sequence = SymbolicSequenceBatch(
        agent_position=agent_position,
        held_entity_slot=held_entity_slot,
        entity_positions=entity_positions,
        entity_kinds=entity_kinds,
        contained_in_slot=contained_in_slot,
        entity_is_open=entity_is_open,
        entity_mask=entity_mask,
        action_kind=action_kind,
        action_target_entity_slot=action_target,
        action_container_slot=action_container,
        action_destination=action_destination,
        action_destination_mask=action_destination_mask,
        step_mask=step_mask,
    )
    query_batch = QueryBatch(
        kind=torch.tensor(query_kind, device=device, dtype=torch.long),
        target_entity_slot=torch.tensor(query_entity, device=device, dtype=torch.long),
        target_field=torch.tensor(query_field, device=device, dtype=torch.long),
        lag=torch.tensor(query_lag, device=device, dtype=torch.long),
        causal_target_kind=torch.tensor(causal_kind, device=device, dtype=torch.long),
        causal_target_entity_slot=torch.tensor(
            causal_entity, device=device, dtype=torch.long
        ),
        causal_value_token=torch.tensor(causal_value, device=device, dtype=torch.long),
    )
    result = CMWMBatch(
        sequence=sequence,
        query=query_batch,
        ask_steps=torch.tensor(ask_steps, device=device, dtype=torch.long),
        reference_steps=torch.tensor(reference_steps, device=device, dtype=torch.long),
        oracle_boundaries=oracle_boundaries,
        answer_targets=torch.tensor(answer_targets, device=device, dtype=torch.long),
        causal_source_steps=torch.tensor(
            causal_source_steps, device=device, dtype=torch.long
        ),
        query_ids=tuple(query_ids),
        episode_ids=tuple(episode_ids),
        answer_vocab_size=answer_vocabulary.size,
    )
    result.validate()
    return result, answer_vocabulary
