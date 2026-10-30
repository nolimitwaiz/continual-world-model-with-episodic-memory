"""Fixed-width Sherlock retrieval primitives, with no encoder or data execution.

The future cluster encoder receives text only. Human scene IDs enter grouping
and scoring, never query/atomic vectors. Every grouping pools the same atomic
vectors; chunk-dependent text truncation and re-encoding are avoided.
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass
from typing import Mapping, Sequence

from cmwm.chen.sherlock_graph_free import (
    AtomicSegment, MemoryRecord, RecallQuery, build_memory_records, segment_iou,
)


DEVELOPMENT_SUBJECTS = frozenset(
    ("s17", "s8", "s3", "s16", "s12", "s1", "s9", "s13", "s4", "s2", "s15", "s7")
)


@dataclass(frozen=True)
class DenseRecord:
    record_id: str
    start_segment: int
    end_segment: int
    vector: tuple[float, ...]

    @property
    def logical_payload_bytes(self) -> int:
        # Float32 vector plus two uint32 bounds; not Python heap/RSS or encoder RAM.
        return 4 * len(self.vector) + 8


def normalized_float32(values: Sequence[float]) -> tuple[float, ...]:
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("vectors must be nonempty and finite")
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("vector has zero or nonfinite norm")
    packed = array("f", (value / norm for value in values))
    if packed.itemsize != 4:
        raise RuntimeError("float32 payload unavailable on this runtime")
    return tuple(packed)


def encoder_text_inputs(
    segments: Sequence[AtomicSegment], queries: Sequence[RecallQuery]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """A text-only boundary: no scene/subject/query IDs are embedded."""
    _validate_development_queries(queries)
    if not segments or any(not s.text.strip() for s in segments):
        raise ValueError("atomic encoding text cannot be empty")
    return tuple(s.text for s in segments), tuple(q.text for q in queries)


def _validate_development_queries(queries: Sequence[RecallQuery]) -> None:
    if not queries or len({q.query_id for q in queries}) != len(queries):
        raise ValueError("queries must be nonempty with unique IDs")
    if any(q.split != "development" or q.source_subject_id not in DEVELOPMENT_SUBJECTS for q in queries):
        raise PermissionError("only the frozen Sherlock development subjects are permitted")
    if any(not q.text.strip() for q in queries):
        raise ValueError("recall text cannot be empty")


def build_dense_records(
    segments: Sequence[AtomicSegment],
    atomic_vectors: Mapping[int, Sequence[float]],
    *,
    condition: str,
    expected_records: int = 50,
    fixed_size: int = 20,
    random_seed: int = 11,
) -> tuple[DenseRecord, ...]:
    """Equal record count/dimension, complete coverage, no stored source text."""
    if condition not in {"fixed_chunks", "random_matched", "human_events"}:
        raise ValueError("fine units are not part of the equal-record primary comparison")
    if [s.segment_number for s in segments] != list(range(1, len(segments) + 1)):
        raise ValueError("atomic segment IDs must be contiguous, ordered and unique")
    if set(atomic_vectors) != {s.segment_number for s in segments}:
        raise ValueError("atomic vector coverage mismatch")
    vectors = {key: normalized_float32(value) for key, value in atomic_vectors.items()}
    dimensions = {len(value) for value in vectors.values()}
    if len(dimensions) != 1:
        raise ValueError("atomic embedding dimensions disagree")
    groups = build_memory_records(
        segments, condition=condition, fixed_size=fixed_size, random_seed=random_seed
    )
    if len(groups) != expected_records:
        raise ValueError("record-count mismatch; cannot claim matched storage")
    result = []
    dimension = next(iter(dimensions))
    for record in groups:
        ids = range(record.start_segment, record.end_segment + 1)
        pooled = normalized_float32(tuple(
            math.fsum(vectors[key][axis] for key in ids) / len(ids)
            for axis in range(dimension)
        ))
        result.append(DenseRecord(record.record_id, record.start_segment, record.end_segment, pooled))
    return tuple(result)


def rank_dense(records: Sequence[DenseRecord], query_vector: Sequence[float]) -> tuple[int, ...]:
    if not records or len({r.record_id for r in records}) != len(records):
        raise ValueError("records must be nonempty and unique")
    query = normalized_float32(query_vector)
    if any(len(r.vector) != len(query) or any(not math.isfinite(x) for x in r.vector) for r in records):
        raise ValueError("record/query vectors must be finite and dimension matched")
    if any(not math.isclose(math.hypot(*r.vector), 1., abs_tol=2e-6) for r in records):
        raise ValueError("record vectors must be unit normalized before cosine ranking")
    scores = [math.fsum(a * b for a, b in zip(query, r.vector)) for r in records]
    # No scene relevance is consulted to break ties.
    return tuple(sorted(range(len(records)), key=lambda i: (-scores[i], records[i].start_segment)))


def evaluate_dense_development(
    records: Sequence[DenseRecord], queries: Sequence[RecallQuery],
    query_vectors: Mapping[str, Sequence[float]], scene_spans: Mapping[int, tuple[int, int]],
    *, cutoff: int = 5,
) -> dict:
    """Development-only scoring, with alignment and returned-span diagnostics.

    Scene-matched metrics can structurally favor human scene records. This
    follow-up is not independent evidence of human recall prediction.
    """
    _validate_development_queries(queries)
    if type(cutoff) is not int or cutoff <= 0 or cutoff > len(records):
        raise ValueError("retrieval cutoff must fit the memory")
    if set(query_vectors) != {q.query_id for q in queries}:
        raise ValueError("query vector identities disagree")
    dimensions = {len(r.vector) for r in records}
    if len(dimensions) != 1:
        raise ValueError("records must have fixed dimension")
    ordered = sorted(records, key=lambda r: r.start_segment)
    if ordered[0].start_segment != 1 or any(r.end_segment < r.start_segment for r in ordered):
        raise ValueError("invalid record spans")
    if any(a.end_segment + 1 != b.start_segment for a, b in zip(ordered, ordered[1:])):
        raise ValueError("records must partition the same atomic sequence")
    rows = []
    for query in queries:
        span = scene_spans[query.scene_number]
        if not 1 <= span[0] <= span[1] <= ordered[-1].end_segment:
            raise ValueError("target span lies outside the atomic sequence")
        ranking = rank_dense(records, query_vectors[query.query_id])
        relevance = [segment_iou(
            MemoryRecord(r.record_id, "dense", r.start_segment, r.end_segment, "", frozenset()), span
        ) for r in records]
        target = min(range(len(records)), key=lambda i: (-relevance[i], records[i].start_segment))
        selected = ranking[:cutoff]
        dcg = sum(relevance[i] / math.log2(rank + 2) for rank, i in enumerate(selected))
        ideal = sum(v / math.log2(rank + 2) for rank, v in enumerate(sorted(relevance, reverse=True)[:cutoff]))
        rows.append({
            "query_id": query.query_id, "subject": query.source_subject_id,
            "best_iou_hit": int(target in selected), "iou_ndcg": dcg / ideal if ideal else 0.,
            "oracle_max_iou": max(relevance),
            "retrieved_record_ids": [records[i].record_id for i in selected],
            "returned_atomic_units": sum(records[i].end_segment - records[i].start_segment + 1 for i in selected),
        })
    subjects = sorted({row["subject"] for row in rows})
    per_subject = {s: math.fsum(row["best_iou_hit"] for row in rows if row["subject"] == s)
                   / sum(row["subject"] == s for row in rows) for s in subjects}
    return {
        "claim_eligible": False, "split": "development", "sealed_test_accessed": False,
        "subject_macro_best_iou_hit": math.fsum(per_subject.values()) / len(per_subject),
        "per_subject_best_iou_hit": per_subject,
        "record_count": len(records), "dimension": next(iter(dimensions)),
        "logical_payload_bytes": sum(r.logical_payload_bytes for r in records),
        "retrieved_records": cutoff, "rows": rows,
        "limitations": [
            "Logical vector/span bytes exclude encoder, build caches, runtime heap and RSS.",
            "Equal vector read count does not imply equal source-text or time-span coverage.",
            "Human scene labels define both one grouping and scoring targets; alignment bias remains.",
            "No fMRI, learned world dynamics, causal reasoning or human memory prediction is tested.",
        ],
    }
