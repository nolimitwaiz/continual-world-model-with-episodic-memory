"""Structured long-delay queries with exact oracle answers."""

from __future__ import annotations

from hashlib import sha256
from typing import Iterable

from cmwm.schemas import (
    ActionKind,
    EntityKind,
    Episode,
    EventKind,
    QueryExample,
    QueryKind,
    StructuredQuery,
    WorldState,
)


def _object_ids(episode: Episode) -> tuple[str, ...]:
    object_ids = sorted(
        entity.entity_id
        for entity in episode.initial_state.entities
        if entity.kind is EntityKind.OBJECT
    )
    if not object_ids:
        raise ValueError("CMWM queries require at least one object")
    return tuple(object_ids)


def state_at(episode: Episode, step: int) -> WorldState:
    if step == 0:
        return episode.initial_state
    if step < 0 or step > len(episode.steps):
        raise ValueError("query step is outside the episode")
    return episode.steps[step - 1].after


def query_memory_reference_step(episode: Episode, example: QueryExample) -> int:
    """Return the hidden record whose delay a query actually tests.

    Causal queries deliberately set ``reference_step`` to ``ask_step`` so the
    model is not told which action caused the outcome.  Audits and evaluation
    still need the oracle source step to stratify the example correctly.
    """

    query = example.query
    if query.kind is not QueryKind.CAUSAL_PREDECESSOR:
        return query.reference_step
    if not isinstance(example.oracle_answer, str):
        raise ValueError("causal answer must be an action source identifier")
    source_steps = [
        record.step_index
        for record in episode.steps
        if any(edge.source == example.oracle_answer for edge in record.causal_edges)
    ]
    if len(source_steps) != 1:
        raise ValueError("causal source must occur in exactly one episode step")
    return source_steps[0]


def effective_query_lag(episode: Episode, example: QueryExample) -> int:
    """Return the true memory delay without changing the model-visible query."""

    lag = example.query.ask_step - query_memory_reference_step(episode, example)
    if lag < 0:
        raise ValueError("effective query lag cannot be negative")
    return lag


def answer_query(episode: Episode, query: StructuredQuery) -> object:
    """Compute an answer without learned components or future-data shortcuts."""

    if query.ask_step > len(episode.steps) or query.reference_step > query.ask_step:
        raise ValueError("query timing is invalid")
    if query.kind is QueryKind.CAUSAL_PREDECESSOR:
        if not query.causal_target:
            raise ValueError("causal query requires causal_target")
        candidates = [
            edge.source
            for record in episode.steps[: query.ask_step]
            for edge in record.causal_edges
            if edge.target == query.causal_target
            and edge.evidence.factual_value == query.causal_value
        ]
        if len(candidates) != 1:
            raise ValueError("causal outcome must resolve to exactly one direct source")
        return candidates[0]

    if not query.target_entity_id or not query.target_field:
        raise ValueError("state query requires target_entity_id and target_field")
    entity = state_at(episode, query.reference_step).entity(query.target_entity_id)
    if query.target_field == "position":
        return entity.position.as_tuple()
    if query.target_field == "contained_in":
        return entity.contained_in
    if query.target_field == "is_open":
        return entity.is_open
    raise ValueError(f"unsupported target field: {query.target_field}")


def generate_queries(
    episode: Episode, *, lags: Iterable[int] = (16, 32, 64, 128)
) -> tuple[QueryExample, ...]:
    """Create deterministic exact-answer queries at the final episode step."""

    ask_step = len(episode.steps)
    requested_lags = tuple(sorted(set(int(value) for value in lags)))
    examples: list[QueryExample] = []
    object_ids = _object_ids(episode)
    for object_id in object_ids:
        object_tag = sha256(object_id.encode("utf-8")).hexdigest()[:8]
        object_suffix = f":{object_tag}" if len(object_ids) > 1 else ""
        for lag in requested_lags:
            if lag <= 0 or lag > ask_step:
                continue
            reference_step = ask_step - lag
            for kind, field in (
                (QueryKind.DELAYED_STATE, "contained_in"),
                (QueryKind.OBJECT_LOCATION_HISTORY, "position"),
            ):
                query = StructuredQuery(
                    query_id=(
                        f"{episode.episode_id}:{kind.value}:{lag}{object_suffix}"
                    ),
                    kind=kind,
                    ask_step=ask_step,
                    reference_step=reference_step,
                    target_entity_id=object_id,
                    target_field=field,
                )
                answer = answer_query(episode, query)
                current_entity = state_at(episode, ask_step).entity(object_id)
                current_value = (
                    current_entity.position.as_tuple()
                    if field == "position"
                    else current_entity.contained_in
                )
                # A query whose historical answer equals the current state can
                # be solved without memory and is excluded.
                if answer != current_value:
                    examples.append(QueryExample(query, answer))

        hidden_steps = [
            record.step_index
            for record in episode.steps
            if record.action.kind is ActionKind.CLOSE_CONTAINER
            and record.after.entity(object_id).contained_in
            == record.action.target_id
            and any(
                event.kind is EventKind.ENTITY_BECAME_HIDDEN
                and event.participants == (object_id,)
                for event in record.events
            )
        ]
        for reference_step in hidden_steps:
            query = StructuredQuery(
                query_id=(
                    f"{episode.episode_id}:occlusion:{reference_step}{object_suffix}"
                ),
                kind=QueryKind.OCCLUSION_CONTINUITY,
                ask_step=ask_step,
                reference_step=reference_step,
                target_entity_id=object_id,
                target_field="contained_in",
            )
            answer = answer_query(episode, query)
            if answer != state_at(episode, ask_step).entity(object_id).contained_in:
                examples.append(QueryExample(query, answer))

    causal_outcomes: dict[tuple[str, str], list[tuple[int, object]]] = {}
    for record in episode.steps:
        for edge in record.causal_edges:
            key = (edge.target, repr(edge.evidence.factual_value))
            causal_outcomes.setdefault(key, []).append((record.step_index, edge))
    for (target, value_key), occurrences in sorted(causal_outcomes.items()):
        if len(occurrences) != 1:
            continue
        source_step, edge = occurrences[0]
        if ask_step - source_step < 16:
            continue
        if (
            episode.split_metadata.template_id.startswith("confusable_")
            and ask_step - source_step not in requested_lags
        ):
            continue
        opaque_id = sha256(
            f"{episode.episode_id}|{target}|{value_key}".encode("utf-8")
        ).hexdigest()[:16]
        query = StructuredQuery(
            query_id=f"{episode.episode_id}:causal:{opaque_id}",
            kind=QueryKind.CAUSAL_PREDECESSOR,
            ask_step=ask_step,
            # The query concerns a uniquely described outcome, not the exact
            # record containing its source action.
            reference_step=ask_step,
            causal_target=target,
            causal_value=edge.evidence.factual_value,
        )
        examples.append(QueryExample(query, answer_query(episode, query)))

    unfiltered = tuple(examples)
    # Causal examples are eligible only when their answer has audit-only
    # distractors matched on every preregistered shortcut field.  This local
    # import avoids making the dependency-free query evaluator depend on audit
    # controls during module initialization.
    from cmwm.synthetic.causal_controls import (
        causal_candidate_controls,
        causal_control_is_matched,
    )

    controls = causal_candidate_controls(episode, unfiltered)
    return tuple(
        example
        for example in unfiltered
        if example.query.kind is not QueryKind.CAUSAL_PREDECESSOR
        or causal_control_is_matched(
            controls.get(example.query.query_id, ()), str(example.oracle_answer)
        )
    )
