"""Audit-only matched distractors for causal-predecessor queries."""

from __future__ import annotations

from cmwm.schemas import Episode, QueryExample, QueryKind
from cmwm.synthetic.audit import CausalCandidateControl


def _target_field(target: str) -> str:
    return target.rsplit(".", 1)[-1]


def _target_entity(target: str) -> str | None:
    pieces = target.split(".")
    return pieces[1] if len(pieces) == 3 and pieces[0] == "entity" else None


def causal_candidate_controls(
    episode: Episode, examples: tuple[QueryExample, ...]
) -> dict[str, tuple[CausalCandidateControl, ...]]:
    """Build candidate controls without exposing them to the model.

    Recency is a preregistered 64-step band rather than an exact timestamp.
    Semantic similarity is a coarse structural score: half for affecting the
    same state-field type and half for matching entity-overlap status.  These
    definitions are fixed for every candidate and prevent exact time, entity,
    target-field, or broad action-semantic shortcuts from uniquely identifying
    the answer.
    """

    source_records: dict[str, tuple[object, tuple[object, ...]]] = {}
    for record in episode.steps:
        grouped: dict[str, list[object]] = {}
        for edge in record.causal_edges:
            grouped.setdefault(edge.source, []).append(edge)
        for source, edges in grouped.items():
            source_records[source] = (record, tuple(edges))

    result: dict[str, tuple[CausalCandidateControl, ...]] = {}
    for example in examples:
        query = example.query
        if query.kind is not QueryKind.CAUSAL_PREDECESSOR:
            continue
        if query.causal_target is None:
            raise ValueError("causal query target is missing")
        query_field = _target_field(query.causal_target)
        query_entity = _target_entity(query.causal_target)
        candidates: list[CausalCandidateControl] = []
        for source, (record, edges) in sorted(source_records.items()):
            if record.step_index > query.ask_step:
                continue
            matching_field = next(
                (
                    edge
                    for edge in edges
                    if _target_field(edge.target) == query_field
                ),
                edges[0],
            )
            candidate_field = _target_field(matching_field.target)
            action_entities = {
                value
                for value in (record.action.target_id, record.action.container_id)
                if value is not None
            }
            entity_overlap = float(
                query_entity is not None and query_entity in action_entities
            )
            answer_overlap = float(
                query_entity is not None
                and query_entity
                in {
                    value
                    for value in (
                        source_records[str(example.oracle_answer)][0].action.target_id,
                        source_records[str(example.oracle_answer)][0].action.container_id,
                    )
                    if value is not None
                }
            )
            semantic_similarity = 0.5 * float(candidate_field == query_field)
            semantic_similarity += 0.5 * float(entity_overlap == answer_overlap)
            candidates.append(
                CausalCandidateControl(
                    source=source,
                    recency=(query.ask_step - record.step_index) // 64,
                    entity_overlap=entity_overlap,
                    target_field=candidate_field,
                    semantic_similarity=semantic_similarity,
                )
            )
        result[query.query_id] = tuple(candidates)
    return result


def causal_control_is_matched(
    candidates: tuple[CausalCandidateControl, ...], answer: str
) -> bool:
    answer_items = [candidate for candidate in candidates if candidate.source == answer]
    if len(answer_items) != 1 or len(candidates) < 2:
        return False
    answer_item = answer_items[0]
    for field, tolerance in (
        ("recency", 0.0),
        ("entity_overlap", 0.0),
        ("target_field", 0.0),
        ("semantic_similarity", 0.05),
    ):
        answer_value = getattr(answer_item, field)
        matched = False
        for candidate in candidates:
            if candidate.source == answer:
                continue
            candidate_value = getattr(candidate, field)
            if isinstance(answer_value, (int, float)) and isinstance(
                candidate_value, (int, float)
            ):
                matched = abs(float(answer_value) - float(candidate_value)) <= tolerance
            else:
                matched = candidate_value == answer_value
            if matched:
                break
        if not matched:
            return False
    return True


def causal_controls_for_dataset(
    episodes: tuple[Episode, ...],
    queries_by_episode: dict[str, tuple[QueryExample, ...]],
) -> dict[str, tuple[CausalCandidateControl, ...]]:
    controls: dict[str, tuple[CausalCandidateControl, ...]] = {}
    for episode in episodes:
        episode_controls = causal_candidate_controls(
            episode, queries_by_episode.get(episode.episode_id, ())
        )
        overlap = set(controls).intersection(episode_controls)
        if overlap:
            raise ValueError("causal query IDs must be globally unique")
        controls.update(episode_controls)
    return controls
