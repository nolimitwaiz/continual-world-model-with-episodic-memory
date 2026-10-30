from __future__ import annotations

import pytest

from cmwm.models.memory import FIXED_METADATA_BYTES, FifoVectorMemory, MemoryRecord


def _record(index: int, key: tuple[float, float]) -> MemoryRecord:
    return MemoryRecord(
        record_id=f"record-{index}",
        key=key,
        value=(float(index), 0.0),
        start_step=index,
        end_step=index,
        boundary_source="test",
    )


def test_fifo_eviction_uses_measured_record_bytes() -> None:
    first = _record(1, (1.0, 0.0))
    second = _record(2, (0.0, 1.0))
    memory = FifoVectorMemory(capacity_bytes=first.storage_bytes, retrieval_top_k=1)

    memory.write(first)
    memory.write(second)

    assert memory.records == (second,)
    assert memory.occupied_bytes == second.storage_bytes


def test_audit_string_lengths_cannot_change_protocol_capacity() -> None:
    short = _record(1, (1.0, 0.0))
    long = MemoryRecord(
        record_id="record-with-a-much-longer-sidecar-name",
        key=short.key,
        value=short.value,
        start_step=short.start_step,
        end_step=short.end_step,
        boundary_source="a-long-audit-only-boundary-source",
    )

    assert short.storage_bytes == long.storage_bytes
    assert short.storage_bytes == 4 * 4 + FIXED_METADATA_BYTES


def test_retrieval_is_bounded_and_uses_one_shared_similarity_rule() -> None:
    memory = FifoVectorMemory(capacity_bytes=4096, retrieval_top_k=2)
    memory.extend([_record(1, (1.0, 0.0)), _record(2, (0.0, 1.0))])

    assert memory.retrieve((0.9, 0.1), top_k=1)[0].record_id == "record-1"
    with pytest.raises(ValueError, match="budget"):
        memory.retrieve((1.0, 0.0), top_k=3)
