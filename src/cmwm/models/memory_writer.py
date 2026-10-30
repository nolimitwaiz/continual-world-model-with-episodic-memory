"""Shared C1--C5 key/value writer with fixed byte-accounting semantics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cmwm.models.contracts import ParameterAccountable
from cmwm.models.symbolic import SYMBOLIC_STATE_DIM


MEMORY_VECTOR_DIM = 128
MEMORY_RECORD_CAPACITY = 64
MAX_RETRIEVED_RECORDS = 4


@dataclass(frozen=True)
class MemoryRecordLayout:
    """Fixed-width layout used to compare storage by bytes, not labels."""

    key_dim: int = MEMORY_VECTOR_DIM
    value_dim: int = MEMORY_VECTOR_DIM
    scalar_bytes: int = 4
    record_id_bytes: int = 8
    interval_bytes: int = 16
    entity_mask_bytes: int = 8
    boundary_source_bytes: int = 1
    alignment_padding_bytes: int = 7

    def __post_init__(self) -> None:
        if self.key_dim != MEMORY_VECTOR_DIM or self.value_dim != MEMORY_VECTOR_DIM:
            raise ValueError("CMWM-E1 fixes memory keys and values at 128 values each")
        if self.scalar_bytes != 4:
            raise ValueError("CMWM-E1 accounting uses float32-equivalent storage")

    @property
    def vector_bytes(self) -> int:
        return (self.key_dim + self.value_dim) * self.scalar_bytes

    @property
    def metadata_bytes(self) -> int:
        return (
            self.record_id_bytes
            + self.interval_bytes
            + self.entity_mask_bytes
            + self.boundary_source_bytes
            + self.alignment_padding_bytes
        )

    @property
    def record_bytes(self) -> int:
        return self.vector_bytes + self.metadata_bytes

    def capacity_bytes(self, records: int = MEMORY_RECORD_CAPACITY) -> int:
        if records < 0 or records > MEMORY_RECORD_CAPACITY:
            raise ValueError("record count exceeds the preregistered capacity")
        return records * self.record_bytes


@dataclass(frozen=True)
class MemoryWriteBatch:
    keys: Tensor
    values: Tensor


class SharedKeyValueMemoryWriter(ParameterAccountable, nn.Module):
    """Summarize any externally grouped segment through one shared writer.

    C1 supplies length-one segments; C2--C5 supply their respective groupings.
    The writer never receives a condition identifier, boundary source, oracle
    event identifier, query label, outcome label, or causal annotation.
    """

    def __init__(self, *, state_dim: int = SYMBOLIC_STATE_DIM) -> None:
        super().__init__()
        if state_dim != SYMBOLIC_STATE_DIM:
            raise ValueError("CMWM-E1 fixes writer state inputs at 128")
        self.state_dim = state_dim
        self.layout = MemoryRecordLayout()
        self.summary = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.GELU(),
            nn.LayerNorm(state_dim),
        )
        self.key_projection = nn.Linear(state_dim, MEMORY_VECTOR_DIM)
        self.value_projection = nn.Linear(state_dim, MEMORY_VECTOR_DIM)

    def forward(self, segment_states: Tensor, segment_mask: Tensor) -> MemoryWriteBatch:
        if segment_states.ndim != 3 or segment_states.shape[-1] != self.state_dim:
            raise ValueError("segment_states must have shape [batch, steps, 128]")
        if tuple(segment_mask.shape) != tuple(segment_states.shape[:2]):
            raise ValueError("segment_mask must have shape [batch, steps]")
        if segment_mask.dtype is not torch.bool:
            raise TypeError("segment_mask must be boolean")
        lengths = segment_mask.sum(dim=1)
        if torch.any(lengths <= 0):
            raise ValueError("every memory segment must contain at least one state")

        mask = segment_mask.unsqueeze(-1).to(segment_states.dtype)
        mean_state = (segment_states * mask).sum(dim=1)
        mean_state = mean_state / lengths.unsqueeze(-1).to(segment_states.dtype)
        last_indices = lengths - 1
        gather_index = last_indices.view(-1, 1, 1).expand(-1, 1, self.state_dim)
        last_state = segment_states.gather(dim=1, index=gather_index).squeeze(1)
        summary = self.summary(torch.cat((mean_state, last_state), dim=-1))
        keys = F.normalize(self.key_projection(summary), p=2, dim=-1)
        values = self.value_projection(summary)
        return MemoryWriteBatch(keys=keys, values=values)

    def storage_accounting(self, records: int) -> dict[str, int]:
        return {
            "records": records,
            "record_bytes": self.layout.record_bytes,
            "vector_bytes_per_record": self.layout.vector_bytes,
            "metadata_bytes_per_record": self.layout.metadata_bytes,
            "total_bytes": self.layout.capacity_bytes(records),
        }
