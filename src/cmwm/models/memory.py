"""Equal-byte FIFO vector memory shared by every CMWM-E1 memory condition."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, sqrt
from typing import Iterable


FLOAT32_BYTES = 4
INT64_BYTES = 8
FIXED_METADATA_BYTES = 40


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    key: tuple[float, ...]
    value: tuple[float, ...]
    start_step: int
    end_step: int
    entity_mask: int = 0
    boundary_source: str = "audit-hidden"

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("record_id cannot be empty")
        if not self.key or not self.value:
            raise ValueError("memory keys and values cannot be empty")
        if self.start_step < 0 or self.end_step < self.start_step:
            raise ValueError("memory step interval is invalid")
        if self.entity_mask < 0:
            raise ValueError("entity_mask must be non-negative")

    @property
    def storage_bytes(self) -> int:
        """Return protocol bytes, not the host Python object's heap footprint.

        Run artifacts serialize ``record_id`` to a uint64 and
        ``boundary_source`` to a one-byte audit code inside the fixed 40-byte
        metadata layout. Their variable-length Python strings are sidecar
        provenance and cannot change a condition's model-visible capacity.
        """

        vector_bytes = (len(self.key) + len(self.value)) * FLOAT32_BYTES
        return vector_bytes + FIXED_METADATA_BYTES


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have equal dimensions")
    numerator = fsum(a * b for a, b in zip(left, right))
    left_norm = sqrt(fsum(value * value for value in left))
    right_norm = sqrt(fsum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


class FifoVectorMemory:
    """A bounded store whose capacity includes vectors and record metadata."""

    def __init__(self, *, capacity_bytes: int, retrieval_top_k: int = 4) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be positive")
        self.capacity_bytes = capacity_bytes
        self.retrieval_top_k = retrieval_top_k
        self._records: list[MemoryRecord] = []
        self._occupied_bytes = 0

    @property
    def records(self) -> tuple[MemoryRecord, ...]:
        return tuple(self._records)

    @property
    def occupied_bytes(self) -> int:
        return self._occupied_bytes

    def clear(self) -> None:
        self._records.clear()
        self._occupied_bytes = 0

    def write(self, record: MemoryRecord) -> None:
        if record.storage_bytes > self.capacity_bytes:
            raise ValueError("record exceeds the entire memory byte budget")
        while self._records and (
            self._occupied_bytes + record.storage_bytes > self.capacity_bytes
        ):
            evicted = self._records.pop(0)
            self._occupied_bytes -= evicted.storage_bytes
        self._records.append(record)
        self._occupied_bytes += record.storage_bytes

    def extend(self, records: Iterable[MemoryRecord]) -> None:
        for record in records:
            self.write(record)

    def retrieve(
        self, query: tuple[float, ...], *, top_k: int | None = None
    ) -> tuple[MemoryRecord, ...]:
        count = self.retrieval_top_k if top_k is None else top_k
        if count <= 0 or count > self.retrieval_top_k:
            raise ValueError("top_k exceeds the preregistered retrieval budget")
        ranked = sorted(
            enumerate(self._records),
            key=lambda item: (
                cosine_similarity(query, item[1].key),
                item[0],
            ),
            reverse=True,
        )
        return tuple(record for _, record in ranked[:count])
