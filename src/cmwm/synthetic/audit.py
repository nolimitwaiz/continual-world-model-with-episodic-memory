"""Collection-level G0 audits for supplied CMWM episodes and queries.

This module is deliberately read-only: it neither generates data nor writes
artifacts.  The report distinguishes hard G0 failures from warnings and keeps
diagnostic statistics separate from scientific model evaluation.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from random import Random
from typing import Iterable, Mapping

from cmwm.schemas import DatasetSplit, Episode, QueryExample, QueryKind
from cmwm.synthetic.manifest import (
    DatasetManifest,
    ManifestError,
    ManifestSeal,
    build_dataset_manifest,
    canonical_json_bytes,
    seal_manifest,
    trajectory_checksum,
)
from cmwm.synthetic.queries import (
    answer_query,
    effective_query_lag,
    query_memory_reference_step,
    state_at,
)
from cmwm.synthetic.validation import ValidationError, validate_episode


class AuditSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class AuditFinding:
    severity: AuditSeverity
    code: str
    message: str
    episode_ids: tuple[str, ...] = ()
    query_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class NamedCount:
    name: str
    count: int


@dataclass(frozen=True)
class LagCount:
    lag: int
    count: int


@dataclass(frozen=True)
class QueryBalanceCell:
    split: str
    family: str
    task: str
    lag: int
    count: int


@dataclass(frozen=True)
class QueryBalanceSummary:
    total_queries: int
    by_split: tuple[NamedCount, ...]
    by_family: tuple[NamedCount, ...]
    by_task: tuple[NamedCount, ...]
    by_lag: tuple[LagCount, ...]
    cells: tuple[QueryBalanceCell, ...]


@dataclass(frozen=True)
class StoragePressureSummary:
    capacity_bytes: int
    frame_record_bytes: int
    capacity_records: int
    episode_count: int
    episodes_with_eviction: int
    observation_records_written: int
    observation_records_evicted: int
    primary_queries: int
    primary_references_outside_context: int
    primary_references_evicted: int


@dataclass(frozen=True)
class SegmentLengthMatchSummary:
    evidence_supplied: bool
    oracle_segment_count: int
    random_segment_count: int
    oracle_length_counts: tuple[LagCount, ...]
    random_length_counts: tuple[LagCount, ...]
    distributions_match: bool


@dataclass(frozen=True)
class LabelDiagnostic:
    task: str
    examples: int
    unique_labels: int
    dominant_label: str
    dominant_count: int
    dominant_fraction: float
    most_common_labels: tuple[NamedCount, ...]


@dataclass(frozen=True)
class ShortcutDiagnostic:
    task: str
    feature: str
    examples: int
    feature_values: int
    covered_examples: int
    coverage: float
    leave_one_episode_out_majority_accuracy: float
    leave_one_episode_out_feature_accuracy: float
    accuracy_uplift: float
    flagged: bool


@dataclass(frozen=True)
class CausalCandidateControl:
    """Precomputed confound attributes for one causal-answer candidate.

    These values are audit evidence, not model inputs.  ``recency`` uses a
    consistent non-negative step distance chosen by the dataset builder.
    ``entity_overlap`` and ``semantic_similarity`` must use one documented
    scale consistently across all candidates in a manifest.
    """

    source: str
    recency: int | None
    entity_overlap: float | None
    target_field: str | None
    semantic_similarity: float | None


@dataclass(frozen=True)
class G0AuditPolicy:
    required_query_kinds: tuple[QueryKind, ...] = tuple(QueryKind)
    required_lags: tuple[int, ...] = (16, 32, 64, 128)
    required_test_families: tuple[str, ...] = ("HF-A", "HF-B")
    fixed_context_steps: int = 32
    primary_lags: tuple[int, ...] = (64, 128)
    frame_memory_capacity_bytes: int = 68_096
    frame_record_bytes: int = 1_064
    min_label_examples: int = 20
    max_dominant_label_fraction: float = 0.95
    max_reported_labels: int = 20
    min_shortcut_examples: int = 50
    min_shortcut_coverage: float = 0.50
    shortcut_accuracy_threshold: float = 0.95
    shortcut_uplift_threshold: float = 0.10
    causal_semantic_similarity_tolerance: float = 0.05
    require_random_matched_evidence: bool = True
    min_training_templates: int = 3
    require_entity_id_diversity: bool = True
    require_event_timing_diversity: bool = True
    require_hf_a_appearance_role_fields: bool = True
    model_max_entities: int = 8

    def __post_init__(self) -> None:
        if self.min_label_examples < 1:
            raise ValueError("min_label_examples must be positive")
        if self.fixed_context_steps < 1:
            raise ValueError("fixed_context_steps must be positive")
        if self.frame_memory_capacity_bytes < 1:
            raise ValueError("frame_memory_capacity_bytes must be positive")
        if self.frame_record_bytes < 1:
            raise ValueError("frame_record_bytes must be positive")
        if self.frame_record_bytes > self.frame_memory_capacity_bytes:
            raise ValueError("one frame record must fit within the memory byte budget")
        if self.max_reported_labels < 1:
            raise ValueError("max_reported_labels must be positive")
        if self.min_shortcut_examples < 1:
            raise ValueError("min_shortcut_examples must be positive")
        if self.min_training_templates < 0:
            raise ValueError("min_training_templates must be non-negative")
        if self.model_max_entities < 1:
            raise ValueError("model_max_entities must be positive")
        for name, value in (
            ("max_dominant_label_fraction", self.max_dominant_label_fraction),
            ("min_shortcut_coverage", self.min_shortcut_coverage),
            ("shortcut_accuracy_threshold", self.shortcut_accuracy_threshold),
            ("shortcut_uplift_threshold", self.shortcut_uplift_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.causal_semantic_similarity_tolerance < 0:
            raise ValueError("causal_semantic_similarity_tolerance must be non-negative")


@dataclass(frozen=True)
class G0AuditReport:
    episode_count: int
    query_count: int
    findings: tuple[AuditFinding, ...]
    balance: QueryBalanceSummary
    storage_pressure: StoragePressureSummary
    segment_length_match: SegmentLengthMatchSummary
    label_diagnostics: tuple[LabelDiagnostic, ...]
    shortcut_diagnostics: tuple[ShortcutDiagnostic, ...]
    manifest: DatasetManifest | None
    manifest_seal: ManifestSeal | None

    @property
    def passed(self) -> bool:
        return not any(
            finding.severity is AuditSeverity.ERROR for finding in self.findings
        )

    @property
    def errors(self) -> tuple[AuditFinding, ...]:
        return tuple(
            finding
            for finding in self.findings
            if finding.severity is AuditSeverity.ERROR
        )


@dataclass(frozen=True)
class _QueryRow:
    episode_id: str
    task: str
    label: str
    features: tuple[tuple[str, str], ...]


_SHORTCUT_FEATURES = (
    "split",
    "family",
    "template_id",
    "mechanism_signature",
    "sequence_length",
    "lag",
    "ask_step",
    "reference_step",
    "target_field",
    "target_entity_id",
    "target_entity_slot",
    "current_answer_token",
    "causal_target",
    "template_lag",
    "template_lag_target_slot",
)


def _label_key(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _family(episode: Episode) -> str:
    metadata = episode.split_metadata
    return metadata.held_out_family or metadata.dataset_split.value


def _named_counts(counter: Counter[str]) -> tuple[NamedCount, ...]:
    return tuple(NamedCount(name, counter[name]) for name in sorted(counter))


def summarize_query_balance(
    episodes: Iterable[Episode],
    queries_by_episode: Mapping[str, Iterable[QueryExample]],
) -> QueryBalanceSummary:
    """Summarize supplied queries by split, held-out family, task, and lag."""

    episode_index = {episode.episode_id: episode for episode in episodes}
    unknown = set(queries_by_episode) - set(episode_index)
    if unknown:
        raise ValueError(f"queries reference unknown episodes: {', '.join(sorted(unknown))}")

    split_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    lag_counts: Counter[int] = Counter()
    cells: Counter[tuple[str, str, str, int]] = Counter()

    for episode_id, examples in queries_by_episode.items():
        episode = episode_index[episode_id]
        split = episode.split_metadata.dataset_split.value
        family = _family(episode)
        for example in examples:
            query = example.query
            lag = effective_query_lag(episode, example)
            task = query.kind.value
            split_counts[split] += 1
            family_counts[family] += 1
            task_counts[task] += 1
            lag_counts[lag] += 1
            cells[(split, family, task, lag)] += 1

    return QueryBalanceSummary(
        total_queries=sum(task_counts.values()),
        by_split=_named_counts(split_counts),
        by_family=_named_counts(family_counts),
        by_task=_named_counts(task_counts),
        by_lag=tuple(LagCount(lag, lag_counts[lag]) for lag in sorted(lag_counts)),
        cells=tuple(
            QueryBalanceCell(split, family, task, lag, count)
            for (split, family, task, lag), count in sorted(cells.items())
        ),
    )


def summarize_storage_pressure(
    episodes: Iterable[Episode],
    queries_by_episode: Mapping[str, Iterable[QueryExample]],
    *,
    policy: G0AuditPolicy,
) -> StoragePressureSummary:
    """Summarize FIFO occupancy using the declared equal-byte frame layout.

    One record is counted for the initial observation and for every subsequent
    ``StepRecord`` observation.  At query time, FIFO retains the latest records
    that fit in the byte budget.
    """

    episode_index = {episode.episode_id: episode for episode in episodes}
    capacity_records = policy.frame_memory_capacity_bytes // policy.frame_record_bytes
    episodes_with_eviction = 0
    written = 0
    evicted = 0
    primary_queries = 0
    outside_context = 0
    evicted_references = 0

    for episode in episode_index.values():
        episode_records = len(episode.steps) + 1
        written += episode_records
        episode_evicted = max(0, episode_records - capacity_records)
        evicted += episode_evicted
        episodes_with_eviction += int(episode_evicted > 0)

    for episode_id, examples in queries_by_episode.items():
        if episode_id not in episode_index:
            continue
        for example in examples:
            query = example.query
            lag = effective_query_lag(episode_index[episode_id], example)
            if lag not in policy.primary_lags:
                continue
            primary_queries += 1
            outside_context += int(lag > policy.fixed_context_steps)
            records_written_at_ask = query.ask_step + 1
            first_retained_step = max(0, records_written_at_ask - capacity_records)
            reference_step = query_memory_reference_step(
                episode_index[episode_id], example
            )
            evicted_references += int(reference_step < first_retained_step)

    return StoragePressureSummary(
        capacity_bytes=policy.frame_memory_capacity_bytes,
        frame_record_bytes=policy.frame_record_bytes,
        capacity_records=capacity_records,
        episode_count=len(episode_index),
        episodes_with_eviction=episodes_with_eviction,
        observation_records_written=written,
        observation_records_evicted=evicted,
        primary_queries=primary_queries,
        primary_references_outside_context=outside_context,
        primary_references_evicted=evicted_references,
    )


def summarize_segment_length_match(
    oracle_segment_lengths_by_episode: Mapping[str, Iterable[int]] | None,
    random_segment_lengths_by_episode: Mapping[str, Iterable[int]] | None,
) -> SegmentLengthMatchSummary:
    """Compare the complete empirical segment-length histograms."""

    if oracle_segment_lengths_by_episode is None or random_segment_lengths_by_episode is None:
        return SegmentLengthMatchSummary(False, 0, 0, (), (), False)
    oracle_counts: Counter[int] = Counter()
    random_counts: Counter[int] = Counter()
    for lengths in oracle_segment_lengths_by_episode.values():
        oracle_counts.update(int(length) for length in lengths)
    for lengths in random_segment_lengths_by_episode.values():
        random_counts.update(int(length) for length in lengths)
    return SegmentLengthMatchSummary(
        evidence_supplied=True,
        oracle_segment_count=sum(oracle_counts.values()),
        random_segment_count=sum(random_counts.values()),
        oracle_length_counts=tuple(
            LagCount(length, oracle_counts[length]) for length in sorted(oracle_counts)
        ),
        random_length_counts=tuple(
            LagCount(length, random_counts[length]) for length in sorted(random_counts)
        ),
        distributions_match=oracle_counts == random_counts,
    )


def _label_diagnostics(
    rows: Iterable[_QueryRow], policy: G0AuditPolicy
) -> tuple[LabelDiagnostic, ...]:
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_task[row.task][row.label] += 1

    diagnostics = []
    for task in sorted(by_task):
        counts = by_task[task]
        total = sum(counts.values())
        common = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        dominant_label, dominant_count = common[0]
        diagnostics.append(
            LabelDiagnostic(
                task=task,
                examples=total,
                unique_labels=len(counts),
                dominant_label=dominant_label,
                dominant_count=dominant_count,
                dominant_fraction=dominant_count / total,
                most_common_labels=tuple(
                    NamedCount(label, count)
                    for label, count in common[: policy.max_reported_labels]
                ),
            )
        )
    return tuple(diagnostics)


def _majority_label(counts: Counter[str]) -> str:
    return min(counts, key=lambda label: (-counts[label], label))


def _majority_label_without(
    counts: Counter[str],
    excluded: Counter[str],
    ranking: tuple[str, ...],
) -> str | None:
    """Return the exact deterministic majority after subtracting one episode.

    Only labels present in ``excluded`` can change rank.  The best unchanged
    label is therefore the first globally ranked label absent from the excluded
    counter; comparing it with the adjusted excluded labels avoids copying and
    rescanning the full (often source-ID-sized) label counter for every row.
    """

    candidates: list[tuple[str, int]] = []
    for label, removed in excluded.items():
        remaining = counts.get(label, 0) - removed
        if remaining > 0:
            candidates.append((label, remaining))
    for label in ranking:
        if label not in excluded:
            candidates.append((label, counts[label]))
            break
    if not candidates:
        return None
    return min(candidates, key=lambda item: (-item[1], item[0]))[0]


def _shortcut_diagnostics(
    rows: tuple[_QueryRow, ...], policy: G0AuditPolicy
) -> tuple[ShortcutDiagnostic, ...]:
    rows_by_task: dict[str, list[_QueryRow]] = defaultdict(list)
    task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    task_episode_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    feature_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    feature_episode_counts: dict[
        tuple[str, str, str, str], Counter[str]
    ] = defaultdict(Counter)

    for row in rows:
        rows_by_task[row.task].append(row)
        task_counts[row.task][row.label] += 1
        task_episode_counts[(row.task, row.episode_id)][row.label] += 1
        for feature, value in row.features:
            feature_counts[(row.task, feature, value)][row.label] += 1
            feature_episode_counts[
                (row.task, feature, value, row.episode_id)
            ][row.label] += 1

    task_rankings = {
        task: tuple(sorted(counts, key=lambda label: (-counts[label], label)))
        for task, counts in task_counts.items()
    }
    feature_rankings = {
        key: tuple(sorted(counts, key=lambda label: (-counts[label], label)))
        for key, counts in feature_counts.items()
    }
    baseline_cache: dict[tuple[str, str], str | None] = {}
    conditional_cache: dict[tuple[str, str, str, str], str | None] = {}

    diagnostics: list[ShortcutDiagnostic] = []
    for task in sorted(rows_by_task):
        task_rows = rows_by_task[task]
        for feature in _SHORTCUT_FEATURES:
            covered = 0
            baseline_correct = 0
            feature_correct = 0
            feature_values = {
                dict(row.features)[feature] for row in task_rows
            }
            for row in task_rows:
                value = dict(row.features)[feature]
                conditional_key = (task, feature, value, row.episode_id)
                if conditional_key not in conditional_cache:
                    conditional_cache[conditional_key] = _majority_label_without(
                        feature_counts[(task, feature, value)],
                        feature_episode_counts[conditional_key],
                        feature_rankings[(task, feature, value)],
                    )
                baseline_key = (task, row.episode_id)
                if baseline_key not in baseline_cache:
                    baseline_cache[baseline_key] = _majority_label_without(
                        task_counts[task],
                        task_episode_counts[baseline_key],
                        task_rankings[task],
                    )
                conditional = conditional_cache[conditional_key]
                baseline = baseline_cache[baseline_key]
                if conditional is None or baseline is None:
                    continue
                covered += 1
                baseline_correct += int(baseline == row.label)
                feature_correct += int(conditional == row.label)

            coverage = covered / len(task_rows) if task_rows else 0.0
            baseline_accuracy = baseline_correct / covered if covered else 0.0
            feature_accuracy = feature_correct / covered if covered else 0.0
            uplift = feature_accuracy - baseline_accuracy
            flagged = (
                covered >= policy.min_shortcut_examples
                and coverage >= policy.min_shortcut_coverage
                and feature_accuracy >= policy.shortcut_accuracy_threshold
                and uplift >= policy.shortcut_uplift_threshold
            )
            diagnostics.append(
                ShortcutDiagnostic(
                    task=task,
                    feature=feature,
                    examples=len(task_rows),
                    feature_values=len(feature_values),
                    covered_examples=covered,
                    coverage=coverage,
                    leave_one_episode_out_majority_accuracy=baseline_accuracy,
                    leave_one_episode_out_feature_accuracy=feature_accuracy,
                    accuracy_uplift=uplift,
                    flagged=flagged,
                )
            )
    return tuple(diagnostics)


def _query_rows(
    episode_index: Mapping[str, Episode],
    query_collections: Mapping[str, tuple[QueryExample, ...]],
    *,
    model_max_entities: int,
) -> tuple[_QueryRow, ...]:
    def slots_for(episode: Episode) -> dict[str, int]:
        entity_ids = sorted(
            entity.entity_id for entity in episode.initial_state.entities
        )
        if len(entity_ids) > model_max_entities:
            raise ValueError("episode exceeds audited model entity-slot budget")
        material = f"cmwm-slot-v1:{episode.split_metadata.episode_seed}".encode(
            "utf-8"
        )
        seed = int.from_bytes(sha256(material).digest()[:8], "big")
        available = list(range(model_max_entities))
        Random(seed).shuffle(available)
        return dict(zip(entity_ids, available[: len(entity_ids)]))

    def model_label(
        episode: Episode,
        example: QueryExample,
        slots: Mapping[str, int],
    ) -> str:
        query = example.query
        if query.kind is QueryKind.CAUSAL_PREDECESSOR:
            source = str(example.oracle_answer)
            records = [
                record
                for record in episode.steps
                if any(edge.source == source for edge in record.causal_edges)
            ]
            if len(records) != 1:
                raise ValueError("causal source must identify exactly one record")
            action = records[0].action
            return _label_key(
                {
                    "action_kind": action.kind.value,
                    "target_slot": (
                        None if action.target_id is None else slots[action.target_id]
                    ),
                    "container_slot": (
                        None
                        if action.container_id is None
                        else slots[action.container_id]
                    ),
                }
            )
        value = example.oracle_answer
        if query.target_field == "contained_in":
            value = None if value is None else slots[str(value)]
        return _label_key({"field": query.target_field, "value": value})

    rows = []
    for episode_id, examples in query_collections.items():
        episode = episode_index.get(episode_id)
        if episode is None:
            continue
        metadata = episode.split_metadata
        slots = slots_for(episode)
        for example in examples:
            query = example.query
            lag = str(effective_query_lag(episode, example))
            target_slot = (
                "<none>"
                if query.target_entity_id is None
                else str(slots[query.target_entity_id])
            )
            current_answer_token = "<none>"
            if query.target_entity_id is not None and query.target_field is not None:
                current_entity = state_at(episode, query.ask_step).entity(
                    query.target_entity_id
                )
                if query.target_field == "position":
                    current_value: object = current_entity.position.as_tuple()
                elif query.target_field == "contained_in":
                    current_value = current_entity.contained_in
                    if current_value is not None:
                        current_value = slots[str(current_value)]
                else:
                    current_value = current_entity.is_open
                current_answer_token = _label_key(
                    {"field": query.target_field, "value": current_value}
                )
            features = {
                "split": metadata.dataset_split.value,
                "family": _family(episode),
                "template_id": metadata.template_id,
                "mechanism_signature": metadata.mechanism_signature,
                "sequence_length": str(len(episode.steps)),
                "lag": lag,
                "ask_step": str(query.ask_step),
                "reference_step": str(query.reference_step),
                "target_field": query.target_field or "<none>",
                "target_entity_id": query.target_entity_id or "<none>",
                "target_entity_slot": target_slot,
                "current_answer_token": current_answer_token,
                "causal_target": query.causal_target or "<none>",
                "template_lag": f"{metadata.template_id}|{lag}",
                "template_lag_target_slot": (
                    f"{metadata.template_id}|{lag}|{target_slot}"
                ),
            }
            rows.append(
                _QueryRow(
                    episode_id=episode_id,
                    task=query.kind.value,
                    label=model_label(episode, example, slots),
                    features=tuple(sorted(features.items())),
                )
            )
    return tuple(rows)


def _matching_distractor_exists(
    candidates: tuple[CausalCandidateControl, ...],
    answer: CausalCandidateControl,
    field: str,
    tolerance: float = 0.0,
) -> bool:
    answer_value = getattr(answer, field)
    if answer_value is None:
        return False
    for candidate in candidates:
        if candidate.source == answer.source:
            continue
        candidate_value = getattr(candidate, field)
        if candidate_value is None:
            continue
        if isinstance(answer_value, (int, float)) and isinstance(
            candidate_value, (int, float)
        ):
            if abs(float(answer_value) - float(candidate_value)) <= tolerance:
                return True
        elif candidate_value == answer_value:
            return True
    return False


def _audit_causal_queries(
    episode_index: Mapping[str, Episode],
    query_collections: Mapping[str, tuple[QueryExample, ...]],
    causal_candidates_by_query: Mapping[
        str, Iterable[CausalCandidateControl]
    ] | None,
    policy: G0AuditPolicy,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    supplied_candidates = causal_candidates_by_query or {}
    causal_query_ids: set[str] = set()

    for episode_id, examples in query_collections.items():
        episode = episode_index.get(episode_id)
        if episode is None:
            continue
        for example in examples:
            query = example.query
            if query.kind is not QueryKind.CAUSAL_PREDECESSOR:
                continue
            causal_query_ids.add(query.query_id)
            answer = str(example.oracle_answer)

            if answer and answer in query.query_id:
                findings.append(
                    AuditFinding(
                        AuditSeverity.ERROR,
                        "causal_query_id_contains_answer",
                        "causal query ID contains its oracle source string",
                        (episode_id,),
                        (query.query_id,),
                    )
                )

            answer_tokens = set(filter(None, re.split(r"[^A-Za-z0-9]+", answer)))
            if str(query.reference_step) in answer_tokens:
                findings.append(
                    AuditFinding(
                        AuditSeverity.ERROR,
                        "causal_source_encodes_reference_step",
                        "causal source string encodes the query reference step",
                        (episode_id,),
                        (query.query_id,),
                    )
                )

            if 1 <= query.reference_step <= len(episode.steps):
                record = episode.steps[query.reference_step - 1]
                pointed_sources = {
                    edge.source
                    for edge in record.causal_edges
                    if edge.target == query.causal_target
                }
                if answer in pointed_sources:
                    findings.append(
                        AuditFinding(
                            AuditSeverity.ERROR,
                            "causal_reference_step_points_to_answer",
                            (
                                "causal query references the exact record containing its "
                                "oracle edge, so relevance is reducible to direct lookup"
                            ),
                            (episode_id,),
                            (query.query_id,),
                        )
                    )

            candidates = tuple(supplied_candidates.get(query.query_id, ()))
            if not candidates:
                findings.append(
                    AuditFinding(
                        AuditSeverity.ERROR,
                        "causal_candidates_missing",
                        (
                            "causal query has no supplied distractor controls for recency, "
                            "entity overlap, target field, and semantic similarity"
                        ),
                        (episode_id,),
                        (query.query_id,),
                    )
                )
                continue

            answer_candidates = [
                candidate for candidate in candidates if candidate.source == answer
            ]
            if len(answer_candidates) != 1:
                findings.append(
                    AuditFinding(
                        AuditSeverity.ERROR,
                        "causal_candidate_answer_multiplicity",
                        "candidate set must contain the oracle source exactly once",
                        (episode_id,),
                        (query.query_id,),
                    )
                )
                continue
            if len(candidates) < 2:
                findings.append(
                    AuditFinding(
                        AuditSeverity.ERROR,
                        "causal_distractors_missing",
                        "causal candidate set contains no distractor",
                        (episode_id,),
                        (query.query_id,),
                    )
                )
                continue

            answer_candidate = answer_candidates[0]
            controls = (
                ("recency", 0.0),
                ("entity_overlap", 0.0),
                ("target_field", 0.0),
                (
                    "semantic_similarity",
                    policy.causal_semantic_similarity_tolerance,
                ),
            )
            for field, tolerance in controls:
                if not _matching_distractor_exists(
                    candidates, answer_candidate, field, tolerance
                ):
                    findings.append(
                        AuditFinding(
                            AuditSeverity.ERROR,
                            f"causal_{field}_unmatched",
                            (
                                f"oracle causal source has no distractor matched on {field}; "
                                "causal relevance is confounded"
                            ),
                            (episode_id,),
                            (query.query_id,),
                        )
                    )

    unknown_controls = set(supplied_candidates) - causal_query_ids
    if unknown_controls:
        findings.append(
            AuditFinding(
                AuditSeverity.ERROR,
                "causal_controls_reference_unknown_query",
                "causal controls reference unknown or non-causal query IDs",
                query_ids=tuple(sorted(unknown_controls)),
            )
        )
    return findings


def _finding_sort_key(finding: AuditFinding) -> tuple[object, ...]:
    severity_order = {
        AuditSeverity.ERROR: 0,
        AuditSeverity.WARNING: 1,
    }
    return (
        severity_order[finding.severity],
        finding.code,
        finding.episode_ids,
        finding.query_ids,
        finding.message,
    )


def audit_g0(
    episodes: Iterable[Episode],
    queries_by_episode: Mapping[str, Iterable[QueryExample]] | None = None,
    *,
    causal_candidates_by_query: Mapping[
        str, Iterable[CausalCandidateControl]
    ] | None = None,
    oracle_segment_lengths_by_episode: Mapping[str, Iterable[int]] | None = None,
    random_segment_lengths_by_episode: Mapping[str, Iterable[int]] | None = None,
    policy: G0AuditPolicy | None = None,
) -> G0AuditReport:
    """Run deterministic G0 integrity, balance, and shortcut audits."""

    active_policy = policy or G0AuditPolicy()
    episode_list = tuple(episodes)
    query_collections = {
        episode_id: tuple(examples)
        for episode_id, examples in (queries_by_episode or {}).items()
    }
    causal_candidate_collections = {
        query_id: tuple(candidates)
        for query_id, candidates in (causal_candidates_by_query or {}).items()
    }
    findings: list[AuditFinding] = []

    if not episode_list:
        findings.append(
            AuditFinding(AuditSeverity.ERROR, "empty_dataset", "dataset is empty")
        )

    episodes_by_id: dict[str, list[Episode]] = defaultdict(list)
    for episode in episode_list:
        episodes_by_id[episode.episode_id].append(episode)
        try:
            validate_episode(episode)
        except (ValidationError, ValueError) as error:
            findings.append(
                AuditFinding(
                    AuditSeverity.ERROR,
                    "invalid_episode",
                    str(error),
                    (episode.episode_id,),
                )
            )

    for episode_id, duplicates in episodes_by_id.items():
        if len(duplicates) > 1:
            findings.append(
                AuditFinding(
                    AuditSeverity.ERROR,
                    "duplicate_episode_id",
                    f"episode ID occurs {len(duplicates)} times",
                    (episode_id,),
                )
            )
    episode_index = {episode_id: items[0] for episode_id, items in episodes_by_id.items()}

    entity_id_signatures = {
        tuple(sorted(entity.entity_id for entity in episode.initial_state.entities))
        for episode in episode_list
    }
    if (
        active_policy.require_entity_id_diversity
        and len(episode_list) >= 2
        and len(entity_id_signatures) < 2
    ):
        findings.append(
            AuditFinding(
                AuditSeverity.ERROR,
                "fixed_entity_id_shortcut",
                "all supplied episodes use the same entity-ID signature",
            )
        )

    training_templates = {
        episode.split_metadata.template_id
        for episode in episode_list
        if episode.split_metadata.dataset_split is DatasetSplit.TRAIN
    }
    if len(training_templates) < active_policy.min_training_templates:
        findings.append(
            AuditFinding(
                AuditSeverity.ERROR,
                "insufficient_training_template_diversity",
                (
                    f"training supplies {len(training_templates)} templates; "
                    f"policy requires at least {active_policy.min_training_templates}"
                ),
            )
        )

    timing_by_template: dict[str, set[tuple[tuple[int, str], ...]]] = defaultdict(set)
    episodes_per_template: Counter[str] = Counter()
    for episode in episode_list:
        template = episode.split_metadata.template_id
        episodes_per_template[template] += 1
        timing_by_template[template].add(
            tuple(
                (record.step_index, record.action.kind.value)
                for record in episode.steps
                if record.action.kind.value != "wait"
            )
        )
    if active_policy.require_event_timing_diversity:
        for template, count in episodes_per_template.items():
            if count >= 2 and len(timing_by_template[template]) == 1:
                findings.append(
                    AuditFinding(
                        AuditSeverity.ERROR,
                        "deterministic_event_timing_shortcut",
                        (
                            f"template {template} has one non-wait action timing "
                            f"signature across {count} episodes"
                        ),
                    )
                )

    hf_a_episodes = [
        episode
        for episode in episode_list
        if episode.split_metadata.held_out_family == "HF-A"
    ]
    if active_policy.require_hf_a_appearance_role_fields and hf_a_episodes:
        required_fields = ("color", "shape", "role")
        hf_a_entities = tuple(
            entity
            for episode in hf_a_episodes
            for entity in episode.initial_state.entities
        )
        missing_fields = [
            field
            for field in required_fields
            if not hf_a_entities
            or not all(
                isinstance(getattr(entity, field, None), str)
                and getattr(entity, field).strip()
                and getattr(entity, field).strip().lower() != "unspecified"
                for entity in hf_a_entities
            )
        ]
        if missing_fields:
            findings.append(
                AuditFinding(
                    AuditSeverity.ERROR,
                    "hf_a_appearance_role_scaffold_only",
                    (
                        "HF-A claims appearance/role shift but entity state lacks: "
                        f"{', '.join(missing_fields)}"
                    ),
                    tuple(sorted(episode.episode_id for episode in hf_a_episodes)),
                )
            )
    if episode_list and not any(
        "collision" in event.kind.value
        for episode in episode_list
        for record in episode.steps
        for event in record.events
    ):
        findings.append(
            AuditFinding(
                AuditSeverity.WARNING,
                "collision_mechanics_not_evidenced",
                "supplied episodes contain no collision event; collision capability is unverified",
            )
        )

    seeds: dict[int, list[Episode]] = defaultdict(list)
    trajectories: dict[str, list[Episode]] = defaultdict(list)
    policy_declarations: set[tuple[object, ...]] = set()
    templates_by_split: dict[DatasetSplit, set[str]] = defaultdict(set)
    for episode in episode_list:
        metadata = episode.split_metadata
        seeds[metadata.episode_seed].append(episode)
        trajectories[trajectory_checksum(episode)].append(episode)
        templates_by_split[metadata.dataset_split].add(metadata.template_id)

        training_mechanisms = set(metadata.training_mechanisms)
        held_out_mechanisms = set(metadata.held_out_mechanisms)
        declared_mechanisms = set(metadata.episode_mechanisms)
        actual_mechanisms = {
            edge.mechanism for record in episode.steps for edge in record.causal_edges
        }
        policy_declarations.add(
            (
                metadata.policy_id,
                tuple(sorted(mechanism.value for mechanism in training_mechanisms)),
                tuple(sorted(mechanism.value for mechanism in held_out_mechanisms)),
            )
        )

        if training_mechanisms & held_out_mechanisms:
            findings.append(
                AuditFinding(
                    AuditSeverity.ERROR,
                    "mechanism_policy_overlap",
                    "training and held-out mechanism declarations overlap",
                    (episode.episode_id,),
                )
            )
        if declared_mechanisms != actual_mechanisms:
            findings.append(
                AuditFinding(
                    AuditSeverity.ERROR,
                    "mechanism_metadata_mismatch",
                    "episode mechanism metadata does not match its causal edges",
                    (episode.episode_id,),
                )
            )
        if (
            metadata.dataset_split is not DatasetSplit.TEST
            and actual_mechanisms & held_out_mechanisms
        ):
            leaked = ", ".join(
                sorted(mechanism.value for mechanism in actual_mechanisms & held_out_mechanisms)
            )
            findings.append(
                AuditFinding(
                    AuditSeverity.ERROR,
                    "held_out_mechanism_leakage",
                    f"held-out mechanisms occur outside test: {leaked}",
                    (episode.episode_id,),
                )
            )
        expected_signature = "+".join(
            sorted(mechanism.value for mechanism in actual_mechanisms)
        )
        if metadata.mechanism_signature != expected_signature:
            findings.append(
                AuditFinding(
                    AuditSeverity.ERROR,
                    "mechanism_signature_mismatch",
                    "mechanism signature does not match the episode's causal edges",
                    (episode.episode_id,),
                )
            )

    if len(policy_declarations) > 1:
        findings.append(
            AuditFinding(
                AuditSeverity.ERROR,
                "inconsistent_split_policy",
                "episodes contain inconsistent mechanism-split declarations",
            )
        )

    for seed, seed_episodes in seeds.items():
        if len(seed_episodes) <= 1:
            continue
        split_names = {episode.split_metadata.dataset_split.value for episode in seed_episodes}
        code = "seed_cross_split_leakage" if len(split_names) > 1 else "duplicate_seed"
        findings.append(
            AuditFinding(
                AuditSeverity.ERROR,
                code,
                f"episode seed {seed} occurs {len(seed_episodes)} times",
                tuple(sorted(episode.episode_id for episode in seed_episodes)),
            )
        )

    for checksum, checksum_episodes in trajectories.items():
        splits = {episode.split_metadata.dataset_split for episode in checksum_episodes}
        if len(splits) > 1:
            findings.append(
                AuditFinding(
                    AuditSeverity.ERROR,
                    "trajectory_cross_split_leakage",
                    f"trajectory {checksum} occurs in multiple splits",
                    tuple(sorted(episode.episode_id for episode in checksum_episodes)),
                )
            )

    development_templates = templates_by_split[DatasetSplit.TRAIN] | templates_by_split[
        DatasetSplit.VALIDATION
    ]
    leaked_templates = development_templates & templates_by_split[DatasetSplit.TEST]
    if leaked_templates:
        findings.append(
            AuditFinding(
                AuditSeverity.ERROR,
                "test_template_leakage",
                f"test templates occur in development: {', '.join(sorted(leaked_templates))}",
            )
        )

    oracle_lengths = (
        {
            episode_id: tuple(int(length) for length in lengths)
            for episode_id, lengths in oracle_segment_lengths_by_episode.items()
        }
        if oracle_segment_lengths_by_episode is not None
        else None
    )
    random_lengths = (
        {
            episode_id: tuple(int(length) for length in lengths)
            for episode_id, lengths in random_segment_lengths_by_episode.items()
        }
        if random_segment_lengths_by_episode is not None
        else None
    )
    segment_length_match = summarize_segment_length_match(oracle_lengths, random_lengths)
    if active_policy.require_random_matched_evidence and not segment_length_match.evidence_supplied:
        findings.append(
            AuditFinding(
                AuditSeverity.ERROR,
                "random_matched_length_distribution_unverified",
                (
                    "oracle/random segment lengths were not supplied; matching boundary count "
                    "alone does not establish the promised empirical length distribution"
                ),
            )
        )
    if segment_length_match.evidence_supplied:
        assert oracle_lengths is not None
        assert random_lengths is not None
        expected_ids = set(episode_index)
        if set(oracle_lengths) != expected_ids or set(random_lengths) != expected_ids:
            findings.append(
                AuditFinding(
                    AuditSeverity.ERROR,
                    "random_matched_episode_coverage_mismatch",
                    "oracle and random segment evidence must cover every supplied episode",
                )
            )
        for episode_id in sorted(expected_ids & set(oracle_lengths) & set(random_lengths)):
            oracle_values = oracle_lengths[episode_id]
            random_values = random_lengths[episode_id]
            if any(length <= 0 for length in (*oracle_values, *random_values)):
                findings.append(
                    AuditFinding(
                        AuditSeverity.ERROR,
                        "invalid_segment_length",
                        "segment lengths must be positive",
                        (episode_id,),
                    )
                )
            if sum(oracle_values) != len(episode_index[episode_id].steps) or sum(
                random_values
            ) != len(episode_index[episode_id].steps):
                findings.append(
                    AuditFinding(
                        AuditSeverity.ERROR,
                        "segment_length_coverage_mismatch",
                        "segment lengths do not cover the complete episode",
                        (episode_id,),
                    )
                )
            if len(oracle_values) != len(random_values):
                findings.append(
                    AuditFinding(
                        AuditSeverity.ERROR,
                        "random_matched_boundary_count_mismatch",
                        "oracle and random conditions have different segment counts",
                        (episode_id,),
                    )
                )
        if not segment_length_match.distributions_match:
            findings.append(
                AuditFinding(
                    AuditSeverity.ERROR,
                    "random_matched_length_distribution_mismatch",
                    "random segments do not match the oracle empirical length histogram",
                )
            )

    unknown_query_episode_ids = set(query_collections) - set(episode_index)
    if unknown_query_episode_ids:
        findings.append(
            AuditFinding(
                AuditSeverity.ERROR,
                "queries_reference_unknown_episode",
                "query collections reference unknown episode IDs",
                tuple(sorted(unknown_query_episode_ids)),
            )
        )

    seen_query_ids: dict[str, str] = {}
    for episode_id, examples in query_collections.items():
        episode = episode_index.get(episode_id)
        if episode is None:
            continue
        for example in examples:
            query = example.query
            prior_episode = seen_query_ids.get(query.query_id)
            if prior_episode is not None:
                findings.append(
                    AuditFinding(
                        AuditSeverity.ERROR,
                        "duplicate_query_id",
                        f"query ID is reused by episodes {prior_episode} and {episode_id}",
                        tuple(sorted({prior_episode, episode_id})),
                        (query.query_id,),
                    )
                )
            else:
                seen_query_ids[query.query_id] = episode_id
            try:
                expected_answer = answer_query(episode, query)
            except (KeyError, IndexError, ValueError) as error:
                findings.append(
                    AuditFinding(
                        AuditSeverity.ERROR,
                        "invalid_query",
                        str(error),
                        (episode_id,),
                        (query.query_id,),
                    )
                )
                continue
            if expected_answer != example.oracle_answer:
                findings.append(
                    AuditFinding(
                        AuditSeverity.ERROR,
                        "oracle_answer_mismatch",
                        "stored oracle answer does not match exact recomputation",
                        (episode_id,),
                        (query.query_id,),
                    )
                )

    try:
        balance = summarize_query_balance(episode_index.values(), query_collections)
    except ValueError:
        known_queries = {
            episode_id: examples
            for episode_id, examples in query_collections.items()
            if episode_id in episode_index
        }
        balance = summarize_query_balance(episode_index.values(), known_queries)

    if queries_by_episode is not None:
        observed_tasks = {count.name for count in balance.by_task}
        for kind in active_policy.required_query_kinds:
            if kind.value not in observed_tasks:
                findings.append(
                    AuditFinding(
                        AuditSeverity.ERROR,
                        "missing_query_task",
                        f"required query task is absent: {kind.value}",
                    )
                )
        observed_lags = {count.lag for count in balance.by_lag}
        for lag in active_policy.required_lags:
            if lag not in observed_lags:
                findings.append(
                    AuditFinding(
                        AuditSeverity.ERROR,
                        "missing_query_lag",
                        f"required query lag is absent: {lag}",
                    )
                )
        observed_test_families = {
            cell.family for cell in balance.cells if cell.split == DatasetSplit.TEST.value
        }
        for family in active_policy.required_test_families:
            if family not in observed_test_families:
                findings.append(
                    AuditFinding(
                        AuditSeverity.ERROR,
                        "missing_test_family_queries",
                        f"required test family has no supplied queries: {family}",
                    )
                )

    storage_pressure = summarize_storage_pressure(
        episode_index.values(), query_collections, policy=active_policy
    )
    capacity_records = storage_pressure.capacity_records
    for episode in episode_index.values():
        if len(episode.steps) + 1 <= capacity_records:
            findings.append(
                AuditFinding(
                    AuditSeverity.ERROR,
                    "frame_memory_no_storage_pressure",
                    (
                        f"episode writes {len(episode.steps) + 1} frame records, "
                        f"which does not overflow the {capacity_records}-record "
                        "equal-byte FIFO budget"
                    ),
                    (episode.episode_id,),
                )
            )
    for episode_id, examples in query_collections.items():
        if episode_id not in episode_index:
            continue
        for example in examples:
            query = example.query
            lag = effective_query_lag(episode_index[episode_id], example)
            if lag not in active_policy.primary_lags:
                continue
            if lag <= active_policy.fixed_context_steps:
                findings.append(
                    AuditFinding(
                        AuditSeverity.ERROR,
                        "primary_query_inside_fixed_context",
                        (
                            f"primary lag {lag} is not beyond the "
                            f"{active_policy.fixed_context_steps}-step context"
                        ),
                        (episode_id,),
                        (query.query_id,),
                    )
                )
            records_written_at_ask = query.ask_step + 1
            first_retained_step = max(0, records_written_at_ask - capacity_records)
            reference_step = query_memory_reference_step(
                episode_index[episode_id], example
            )
            if reference_step >= first_retained_step:
                findings.append(
                    AuditFinding(
                        AuditSeverity.ERROR,
                        "primary_reference_not_evicted",
                        (
                            f"primary reference step {reference_step} remains in "
                            f"the retained {capacity_records}-record FIFO frame window"
                        ),
                        (episode_id,),
                        (query.query_id,),
                    )
                )

    rows = _query_rows(
        episode_index,
        query_collections,
        model_max_entities=active_policy.model_max_entities,
    )
    label_diagnostics = _label_diagnostics(rows, active_policy)
    for diagnostic in label_diagnostics:
        if diagnostic.examples < active_policy.min_label_examples:
            continue
        if diagnostic.unique_labels == 1:
            findings.append(
                AuditFinding(
                    AuditSeverity.ERROR,
                    "degenerate_label",
                    f"task {diagnostic.task} has only one oracle label",
                )
            )
        elif diagnostic.dominant_fraction > active_policy.max_dominant_label_fraction:
            findings.append(
                AuditFinding(
                    AuditSeverity.ERROR,
                    "dominant_label_shortcut",
                    (
                        f"task {diagnostic.task} has dominant-label fraction "
                        f"{diagnostic.dominant_fraction:.3f}"
                    ),
                )
            )

    shortcut_diagnostics = _shortcut_diagnostics(rows, active_policy)
    for diagnostic in shortcut_diagnostics:
        if diagnostic.flagged:
            findings.append(
                AuditFinding(
                    AuditSeverity.ERROR,
                    "single_field_shortcut",
                    (
                        f"{diagnostic.feature} predicts task {diagnostic.task} at "
                        f"{diagnostic.leave_one_episode_out_feature_accuracy:.3f} "
                        f"accuracy (uplift {diagnostic.accuracy_uplift:.3f}, "
                        f"coverage {diagnostic.coverage:.3f})"
                    ),
                )
            )

    findings.extend(
        _audit_causal_queries(
            episode_index,
            query_collections,
            causal_candidate_collections,
            active_policy,
        )
    )

    manifest: DatasetManifest | None = None
    manifest_seal: ManifestSeal | None = None
    try:
        audit_sidecars: dict[str, object] = {
            "causal_candidates_by_query": causal_candidate_collections,
        }
        if oracle_lengths is not None:
            audit_sidecars["oracle_segment_lengths_by_episode"] = oracle_lengths
        if random_lengths is not None:
            audit_sidecars["random_segment_lengths_by_episode"] = random_lengths
        manifest = build_dataset_manifest(
            episode_list,
            query_collections,
            audit_sidecars=audit_sidecars,
        )
        manifest_seal = seal_manifest(manifest)
    except ManifestError as error:
        findings.append(
            AuditFinding(
                AuditSeverity.ERROR,
                "manifest_unsealable",
                str(error),
            )
        )

    return G0AuditReport(
        episode_count=len(episode_list),
        query_count=balance.total_queries,
        findings=tuple(sorted(findings, key=_finding_sort_key)),
        balance=balance,
        storage_pressure=storage_pressure,
        segment_length_match=segment_length_match,
        label_diagnostics=label_diagnostics,
        shortcut_diagnostics=shortcut_diagnostics,
        manifest=manifest,
        manifest_seal=manifest_seal,
    )
