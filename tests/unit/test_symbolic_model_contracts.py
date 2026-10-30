from __future__ import annotations

from dataclasses import fields
from inspect import signature

import pytest
import torch
from torch.nn import functional as F

from cmwm.models.actions import ActionBatch, CanonicalActionEncoder
from cmwm.models.contracts import finite_mask_value
from cmwm.models.memory_writer import (
    MEMORY_RECORD_CAPACITY,
    MemoryRecordLayout,
    SharedKeyValueMemoryWriter,
)
from cmwm.models.queries import (
    QueryBatch,
    QueryHeadOutput,
    RetrievedMemoryBatch,
    SharedQueryHead,
    StructuredQueryEncoder,
)
from cmwm.models.symbolic import (
    CanonicalSymbolicStateAdapter,
    CanonicalSymbolicStepAdapter,
    SymbolicStateBatch,
)
from cmwm.models.temporal import FixedGRUTemporalCore


def _symbolic_batch() -> SymbolicStateBatch:
    return SymbolicStateBatch(
        agent_position=torch.tensor([[0.0, 0.0], [2.0, 1.0]]),
        held_entity_slot=torch.tensor([-1, 0]),
        entity_positions=torch.tensor(
            [
                [[1.0, 0.0], [3.0, 1.0], [0.0, 0.0]],
                [[2.0, 1.0], [4.0, 2.0], [0.0, 0.0]],
            ]
        ),
        entity_kinds=torch.tensor([[0, 1, 0], [0, 1, 0]]),
        contained_in_slot=torch.tensor([[1, -1, -1], [-1, -1, -1]]),
        entity_is_open=torch.tensor([[False, True, False], [False, False, False]]),
        entity_mask=torch.tensor([[True, True, False], [True, True, False]]),
    )


def _action_batch() -> ActionBatch:
    return ActionBatch(
        kind=torch.tensor([0, 7]),
        target_entity_slot=torch.tensor([-1, 0]),
        container_slot=torch.tensor([-1, 1]),
        destination=torch.tensor([[0.0, 0.0], [4.0, 2.0]]),
        destination_mask=torch.tensor([False, True]),
    )


def test_fixed_symbolic_temporal_query_shapes() -> None:
    adapter = CanonicalSymbolicStateAdapter(
        max_entities=8, grid_width=6, grid_height=4
    )
    state_only = adapter(_symbolic_batch())
    assert state_only.shape == (2, 128)

    step_adapter = CanonicalSymbolicStepAdapter(
        max_entities=8, grid_width=6, grid_height=4
    )
    encoded = step_adapter(_symbolic_batch(), _action_batch())
    assert encoded.shape == (2, 128)

    temporal = FixedGRUTemporalCore()
    temporal_output = temporal(encoded.unsqueeze(1).expand(-1, 5, -1))
    assert temporal_output.sequence.shape == (2, 5, 128)
    assert temporal_output.final_state.shape == (1, 2, 128)

    query = QueryBatch(
        kind=torch.tensor([0, 3]),
        target_entity_slot=torch.tensor([0, 1]),
        target_field=torch.tensor([0, 2]),
        lag=torch.tensor([64, 128]),
        causal_target_kind=torch.tensor([-1, 4]),
        causal_target_entity_slot=torch.tensor([-1, 0]),
        causal_value_token=torch.tensor([-1, 3]),
    )
    query_encoder = StructuredQueryEncoder(
        max_entities=8,
        field_vocab_size=6,
        causal_target_vocab_size=12,
        answer_vocab_size=32,
    )
    query_encoding = query_encoder(query)
    retrieved = RetrievedMemoryBatch(
        keys=torch.randn(2, 4, 128),
        values=torch.randn(2, 4, 128),
        mask=torch.tensor([[True, True, False, False], [True, True, True, True]]),
    )
    output = SharedQueryHead(answer_vocab_size=32)(
        temporal_output.sequence[:, -1], query_encoding, retrieved
    )
    assert isinstance(output, QueryHeadOutput)
    assert output.answer_logits.shape == (2, 32)
    assert output.candidate_logits.shape == (2, 4)


def test_one_writer_contract_handles_frame_and_segment_grouping() -> None:
    writer = SharedKeyValueMemoryWriter()
    frame = torch.randn(2, 1, 128)
    segment = torch.randn(2, 7, 128)
    frame_write = writer(frame, torch.ones(2, 1, dtype=torch.bool))
    segment_write = writer(segment, torch.ones(2, 7, dtype=torch.bool))

    assert frame_write.keys.shape == segment_write.keys.shape == (2, 128)
    assert frame_write.values.shape == segment_write.values.shape == (2, 128)
    assert writer.parameter_accounting().total == sum(
        parameter.numel() for parameter in writer.parameters()
    )


def test_memory_layout_counts_fixed_metadata_and_total_bytes() -> None:
    layout = MemoryRecordLayout()
    assert layout.vector_bytes == (128 + 128) * 4
    assert layout.metadata_bytes > 0
    assert layout.record_bytes == layout.vector_bytes + layout.metadata_bytes
    assert layout.capacity_bytes() == MEMORY_RECORD_CAPACITY * layout.record_bytes


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_masking_sentinel_is_finite_and_representable(dtype: torch.dtype) -> None:
    empty = torch.empty(0, dtype=dtype)
    sentinel = finite_mask_value(empty)
    padded = F.pad(empty, (0, 4), value=sentinel)

    assert padded.dtype is dtype
    assert torch.isfinite(padded).all()
    assert torch.all(padded == torch.finfo(dtype).min)


@pytest.mark.parametrize(
    ("constructor", "kwargs"),
    [
        (
            CanonicalSymbolicStateAdapter,
            {
                "max_entities": 8,
                "grid_width": 6,
                "grid_height": 4,
                "output_dim": 64,
            },
        ),
        (
            CanonicalActionEncoder,
            {
                "max_entities": 8,
                "grid_width": 6,
                "grid_height": 4,
                "output_dim": 64,
            },
        ),
        (
            CanonicalSymbolicStepAdapter,
            {
                "max_entities": 8,
                "grid_width": 6,
                "grid_height": 4,
                "output_dim": 64,
            },
        ),
        (FixedGRUTemporalCore, {"hidden_dim": 64}),
        (SharedKeyValueMemoryWriter, {"state_dim": 64}),
    ],
)
def test_preregistered_dimensions_cannot_drift(constructor: type, kwargs: dict) -> None:
    with pytest.raises(ValueError, match="fixes"):
        constructor(**kwargs)


def test_model_inputs_exclude_condition_and_oracle_channels() -> None:
    forbidden = {
        "condition_id",
        "boundary_source",
        "boundary_label",
        "event_id",
        "causal_edges",
        "counterfactual",
        "effect",
        "oracle_answer",
        "outcome_label",
        "candidate_id",
        "record_id",
    }
    state_fields = {field.name for field in fields(SymbolicStateBatch)}
    action_fields = {field.name for field in fields(ActionBatch)}
    query_fields = {field.name for field in fields(QueryBatch)}
    retrieved_fields = {field.name for field in fields(RetrievedMemoryBatch)}
    assert forbidden.isdisjoint(state_fields)
    assert forbidden.isdisjoint(action_fields)
    assert forbidden.isdisjoint(query_fields)
    assert forbidden.isdisjoint(retrieved_fields)

    assert tuple(signature(CanonicalSymbolicStateAdapter.forward).parameters) == (
        "self",
        "batch",
    )
    assert tuple(signature(SharedKeyValueMemoryWriter.forward).parameters) == (
        "self",
        "segment_states",
        "segment_mask",
    )
    assert tuple(signature(CanonicalSymbolicStepAdapter.forward).parameters) == (
        "self",
        "state",
        "action",
    )
    assert tuple(signature(SharedQueryHead.forward).parameters) == (
        "self",
        "current_state",
        "query_encoding",
        "retrieved",
    )
