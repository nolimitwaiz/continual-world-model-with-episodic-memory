"""Leakage-controlled graph-free retrieval benchmark for Sherlock recall.

The benchmark treats the 1,000 released encoding-label rows as atomic units.
It varies only how consecutive units are grouped into memory records: fine
units, fixed chunks, random chunks matched to human event lengths, or the 50
human scene groups. Participant recall text is a query; its coded scene is used
only by the evaluator.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+")
SCENE_RE = re.compile(r"(\d+)\.\s*(.+)")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


@dataclass(frozen=True)
class AtomicSegment:
    segment_number: int
    scene_number: int
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    condition: str
    start_segment: int
    end_segment: int
    text: str
    scene_numbers: frozenset[int]

    @property
    def token_count(self) -> int:
        return len(tokenize(self.text, remove_stopwords=False))


@dataclass(frozen=True)
class RecallQuery:
    query_id: str
    source_subject_id: str
    transcript_alias: str
    scene_number: int
    text: str
    occurrence_count: int
    split: str


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    values = TOKEN_RE.findall(text.lower())
    return [value for value in values if value not in STOPWORDS] if remove_stopwords else values


def subject_to_transcript_alias(source_subject_id: str) -> str:
    """Map released Princeton `sN` IDs to the timestamp paper's `NNxx` IDs.

    The mapping is supported by Table 1 of the primary timestamp publication
    (PMCID PMC10460947), which maps each NN file to its Princeton subject number.
    It is not inferred from recall content or model output.
    """

    match = re.fullmatch(r"s(\d+)", source_subject_id)
    if not match or not 1 <= int(match.group(1)) <= 17:
        raise ValueError(f"invalid Sherlock source subject: {source_subject_id}")
    return f"NN{int(match.group(1)):02d}"


def load_atomic_segments(
    path: str | Path,
    *,
    expected_segments: int = 1000,
    expected_scenes: int = 50,
) -> list[AtomicSegment]:
    segments: list[AtomicSegment] = []
    current_scene: int | None = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            record = json.loads(raw)
            values = record["values"]
            title = str(values.get("Scene Segments") or "").strip()
            if title:
                match = SCENE_RE.fullmatch(title)
                if not match:
                    raise ValueError(f"unrecognized scene marker at line {line_number}: {title!r}")
                current_scene = int(match.group(1))
            if current_scene is None:
                raise ValueError("atomic labels precede the first scene marker")
            text = str(values.get("Scene Details - A Level") or "").strip()
            segments.append(
                AtomicSegment(
                    segment_number=int(values["Segment Number"]),
                    scene_number=current_scene,
                    start_sec=float(values["Start Time (s)"]),
                    end_sec=float(values["End Time (s)"]),
                    text=text,
                )
            )
    if len(segments) != expected_segments:
        raise ValueError(f"expected {expected_segments} atomic segments, found {len(segments)}")
    if [item.segment_number for item in segments] != list(range(1, expected_segments + 1)):
        raise ValueError("atomic segment numbers must be sequential")
    scenes = [item.scene_number for item in segments]
    if sorted(set(scenes)) != list(range(1, expected_scenes + 1)):
        raise ValueError("scene identifiers must be sequential")
    if any(right < left for left, right in zip(scenes, scenes[1:])):
        raise ValueError("scene identifiers must form contiguous ordered groups")
    if any(item.end_sec < item.start_sec for item in segments):
        raise ValueError("atomic segment durations cannot be negative")
    return segments


def atomic_timing_diagnostics(segments: Sequence[AtomicSegment]) -> dict[str, int]:
    """Expose released timing artifacts without changing the atomic sequence."""

    return {
        "zero_duration_segments": sum(item.end_sec == item.start_sec for item in segments),
        "negative_duration_segments": sum(item.end_sec < item.start_sec for item in segments),
        "start_time_resets": sum(
            right.start_sec < left.start_sec for left, right in zip(segments, segments[1:])
        ),
    }


def scene_spans_from_segments(
    segments: Sequence[AtomicSegment],
) -> dict[int, tuple[int, int]]:
    groups = _contiguous_groups(segments)
    return {
        group[0].scene_number: (group[0].segment_number, group[-1].segment_number)
        for group in groups
    }


def segment_iou(record: MemoryRecord, target_span: tuple[int, int]) -> float:
    target_start, target_end = target_span
    intersection = max(
        0,
        min(record.end_segment, target_end) - max(record.start_segment, target_start) + 1,
    )
    record_length = record.end_segment - record.start_segment + 1
    target_length = target_end - target_start + 1
    union = record_length + target_length - intersection
    return intersection / union if union else 0.0


def _contiguous_groups(segments: Sequence[AtomicSegment]) -> list[list[AtomicSegment]]:
    groups: list[list[AtomicSegment]] = []
    for segment in segments:
        if not groups or groups[-1][-1].scene_number != segment.scene_number:
            groups.append([])
        groups[-1].append(segment)
    return groups


def _records_from_groups(
    groups: Sequence[Sequence[AtomicSegment]], *, condition: str
) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    for index, group in enumerate(groups, start=1):
        if not group:
            raise ValueError("memory groups cannot be empty")
        records.append(
            MemoryRecord(
                record_id=f"{condition}_{index:04d}",
                condition=condition,
                start_segment=group[0].segment_number,
                end_segment=group[-1].segment_number,
                text=" ".join(item.text for item in group if item.text).strip(),
                scene_numbers=frozenset(item.scene_number for item in group),
            )
        )
    return records


def build_memory_records(
    segments: Sequence[AtomicSegment],
    *,
    condition: str,
    random_seed: int = 0,
    fixed_size: int = 20,
) -> list[MemoryRecord]:
    if not segments:
        raise ValueError("segments cannot be empty")
    if condition == "fine_units":
        groups = [[segment] for segment in segments]
    elif condition == "human_events":
        groups = _contiguous_groups(segments)
    elif condition == "fixed_chunks":
        if fixed_size <= 0:
            raise ValueError("fixed_size must be positive")
        groups = [list(segments[start : start + fixed_size]) for start in range(0, len(segments), fixed_size)]
    elif condition == "random_matched":
        lengths = [len(group) for group in _contiguous_groups(segments)]
        random.Random(random_seed).shuffle(lengths)
        groups = []
        start = 0
        for length in lengths:
            groups.append(list(segments[start : start + length]))
            start += length
        if start != len(segments):
            raise AssertionError("matched random groups do not cover all segments")
    else:
        raise ValueError(f"unknown segmentation condition: {condition}")
    return _records_from_groups(groups, condition=condition)


def _load_jsonl(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_recall_queries(
    token_path: str | Path,
    occurrence_path: str | Path,
    *,
    subject_splits: Mapping[str, str],
) -> tuple[list[RecallQuery], dict[str, int]]:
    tokens = _load_jsonl(token_path)
    occurrences = _load_csv(occurrence_path)
    tokens_by_alias: dict[str, list[dict[str, object]]] = defaultdict(list)
    non_string_tokens = 0
    for token in tokens:
        alias = str(token["transcript_alias"])
        tokens_by_alias[alias].append(token)
        if token.get("word_value_type") != "string" or not isinstance(token.get("word_raw"), str):
            non_string_tokens += 1
    for rows in tokens_by_alias.values():
        rows.sort(key=lambda row: int(row["source_row"]))

    grouped: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    empty_occurrences = 0
    for occurrence in occurrences:
        subject = occurrence["source_subject_id"]
        if subject not in subject_splits:
            raise ValueError(f"missing split assignment for {subject}")
        alias = subject_to_transcript_alias(subject)
        scene_number = int(occurrence["scene_number"])
        if not 1 <= scene_number <= 50:
            raise ValueError(f"invalid Sherlock scene number for {subject}: {scene_number}")
        start = float(occurrence["recall_start_sec"])
        end = float(occurrence["recall_end_sec"])
        if end <= start:
            raise ValueError(f"non-positive recall interval for {subject}")
        words = [
            str(row["word_raw"]).strip()
            for row in tokens_by_alias.get(alias, [])
            if row.get("word_value_type") == "string"
            and isinstance(row.get("word_raw"), str)
            and start <= float(row["princeton_word_onset_sec"]) < end
            and str(row["word_raw"]).strip()
        ]
        text = " ".join(words)
        if not text:
            empty_occurrences += 1
        grouped[(subject, scene_number)].append(
            (int(occurrence["recall_sequence_index"]), text)
        )

    queries: list[RecallQuery] = []
    empty_queries = 0
    for (subject, scene), parts in sorted(
        grouped.items(), key=lambda item: (int(item[0][0][1:]), item[0][1])
    ):
        parts.sort(key=lambda item: item[0])
        text = " ".join(value for _, value in parts if value).strip()
        if not text:
            empty_queries += 1
            continue
        queries.append(
            RecallQuery(
                query_id=f"sherlock_{subject}_e{scene:03d}",
                source_subject_id=subject,
                transcript_alias=subject_to_transcript_alias(subject),
                scene_number=scene,
                text=text,
                occurrence_count=len(parts),
                split=subject_splits[subject],
            )
        )
    diagnostics = {
        "raw_token_rows": len(tokens),
        "non_string_token_rows_excluded": non_string_tokens,
        "recall_occurrences": len(occurrences),
        "empty_occurrence_intervals": empty_occurrences,
        "unique_recalled_subject_scenes": len(grouped),
        "empty_aggregated_queries_excluded": empty_queries,
        "queries": len(queries),
    }
    return queries, diagnostics


class BM25Index:
    def __init__(self, records: Sequence[MemoryRecord], *, k1: float = 1.2, b: float = 0.75) -> None:
        if not records:
            raise ValueError("BM25 requires records")
        self.records = list(records)
        self.k1 = float(k1)
        self.b = float(b)
        self.documents = [tokenize(record.text) for record in records]
        self.lengths = [len(document) for document in self.documents]
        self.average_length = sum(self.lengths) / len(self.lengths)
        self.document_frequency: Counter[str] = Counter()
        for document in self.documents:
            self.document_frequency.update(set(document))

    def rank(self, text: str) -> list[tuple[MemoryRecord, float]]:
        query = tokenize(text)
        count = len(self.records)
        scored: list[tuple[MemoryRecord, float]] = []
        for record, document, length in zip(self.records, self.documents, self.lengths):
            frequencies = Counter(document)
            score = 0.0
            for term in query:
                df = self.document_frequency.get(term, 0)
                inverse = math.log(1.0 + (count - df + 0.5) / (df + 0.5))
                tf = frequencies.get(term, 0)
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * length / max(self.average_length, 1.0)
                )
                if denominator:
                    score += inverse * tf * (self.k1 + 1.0) / denominator
            scored.append((record, score))
        return sorted(scored, key=lambda item: (-item[1], item[0].record_id))


def evaluate_bm25(
    records: Sequence[MemoryRecord],
    queries: Iterable[RecallQuery],
    *,
    split: str,
    scene_spans: Mapping[int, tuple[int, int]],
    ks: Sequence[int] = (1, 3, 5, 10),
    k1: float = 1.2,
    b: float = 0.75,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("retrieval cutoffs must be positive")
    selected = [query for query in queries if query.split == split]
    if not selected:
        raise ValueError(f"no queries in split {split}")
    index = BM25Index(records, k1=k1, b=b)
    rows: list[dict[str, object]] = []
    for query in selected:
        ranking = index.rank(query.text)
        if query.scene_number not in scene_spans:
            raise ValueError(f"missing target span for scene {query.scene_number}")
        target_span = scene_spans[query.scene_number]
        alignments = {
            record.record_id: segment_iou(record, target_span) for record in records
        }
        best_aligned_record = sorted(
            records, key=lambda record: (-alignments[record.record_id], record.record_id)
        )[0]
        best_aligned_rank = next(
            rank
            for rank, (record, _) in enumerate(ranking, start=1)
            if record.record_id == best_aligned_record.record_id
        )
        relevant_ranks = [
            rank
            for rank, (record, _) in enumerate(ranking, start=1)
            if query.scene_number in record.scene_numbers
        ]
        first_rank = min(relevant_ranks) if relevant_ranks else None
        row: dict[str, object] = {
            "condition": records[0].condition,
            "query_id": query.query_id,
            "source_subject_id": query.source_subject_id,
            "transcript_alias": query.transcript_alias,
            "target_scene_number": query.scene_number,
            "query_token_count": len(tokenize(query.text, remove_stopwords=False)),
            "target_record_count": len(relevant_ranks),
            "first_relevant_rank": first_rank,
            "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
            "best_aligned_record_id": best_aligned_record.record_id,
            "best_alignment_iou": alignments[best_aligned_record.record_id],
            "best_aligned_rank": best_aligned_rank,
            "best_aligned_reciprocal_rank": 1.0 / best_aligned_rank,
        }
        for k in ks:
            top = ranking[:k]
            row[f"hit_at_{k}"] = int(first_rank is not None and first_rank <= k)
            row[f"best_iou_hit_at_{k}"] = int(best_aligned_rank <= k)
            discounted_gain = sum(
                alignments[record.record_id] / math.log2(rank + 1)
                for rank, (record, _) in enumerate(top, start=1)
            )
            ideal_gain = sum(
                relevance / math.log2(rank + 1)
                for rank, relevance in enumerate(
                    sorted(alignments.values(), reverse=True)[:k], start=1
                )
            )
            row[f"iou_ndcg_at_{k}"] = discounted_gain / ideal_gain if ideal_gain else 0.0
            row[f"returned_tokens_at_{k}"] = sum(record.token_count for record, _ in top)
        rows.append(row)

    subjects = sorted({str(row["source_subject_id"]) for row in rows})
    summary: dict[str, object] = {
        "condition": records[0].condition,
        "split": split,
        "records": len(records),
        "record_tokens_total": sum(record.token_count for record in records),
        "queries": len(rows),
        "subjects": len(subjects),
        "mean_reciprocal_rank": sum(float(row["reciprocal_rank"]) for row in rows) / len(rows),
        "mean_target_record_count": sum(int(row["target_record_count"]) for row in rows) / len(rows),
        "mean_best_alignment_iou": sum(float(row["best_alignment_iou"]) for row in rows)
        / len(rows),
        "mean_best_aligned_reciprocal_rank": sum(
            float(row["best_aligned_reciprocal_rank"]) for row in rows
        )
        / len(rows),
    }
    for k in ks:
        summary[f"micro_hit_at_{k}"] = sum(int(row[f"hit_at_{k}"]) for row in rows) / len(rows)
        per_subject = []
        for subject in subjects:
            subject_rows = [row for row in rows if row["source_subject_id"] == subject]
            per_subject.append(
                sum(int(row[f"hit_at_{k}"]) for row in subject_rows) / len(subject_rows)
            )
        summary[f"subject_macro_hit_at_{k}"] = sum(per_subject) / len(per_subject)
        summary[f"micro_best_iou_hit_at_{k}"] = sum(
            int(row[f"best_iou_hit_at_{k}"]) for row in rows
        ) / len(rows)
        summary[f"micro_iou_ndcg_at_{k}"] = sum(
            float(row[f"iou_ndcg_at_{k}"]) for row in rows
        ) / len(rows)
        per_subject_best_iou = []
        per_subject_iou_ndcg = []
        for subject in subjects:
            subject_rows = [row for row in rows if row["source_subject_id"] == subject]
            per_subject_best_iou.append(
                sum(int(row[f"best_iou_hit_at_{k}"]) for row in subject_rows)
                / len(subject_rows)
            )
            per_subject_iou_ndcg.append(
                sum(float(row[f"iou_ndcg_at_{k}"]) for row in subject_rows)
                / len(subject_rows)
            )
        summary[f"subject_macro_best_iou_hit_at_{k}"] = sum(per_subject_best_iou) / len(
            per_subject_best_iou
        )
        summary[f"subject_macro_iou_ndcg_at_{k}"] = sum(per_subject_iou_ndcg) / len(
            per_subject_iou_ndcg
        )
        summary[f"mean_returned_tokens_at_{k}"] = sum(
            int(row[f"returned_tokens_at_{k}"]) for row in rows
        ) / len(rows)
    return rows, summary
