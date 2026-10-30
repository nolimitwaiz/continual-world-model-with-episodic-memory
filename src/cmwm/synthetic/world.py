"""Deterministic symbolic 2-D world and one-step intervention semantics."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

from cmwm.schemas import (
    CAUSAL_EDGE_METHOD,
    Action,
    ActionKind,
    BoundaryLabel,
    BoundaryReason,
    CausalEdge,
    EntityKind,
    EntityObservation,
    EntityState,
    Event,
    EventKind,
    InterventionEvidence,
    Mechanism,
    Observation,
    Position,
    StepRecord,
    WorldConfig,
    WorldState,
)


class InvalidActionError(ValueError):
    """Raised when an action violates the symbolic world's mechanics."""


_ACTION_MECHANISM: Mapping[ActionKind, Mechanism] = {
    ActionKind.MOVE_AGENT: Mechanism.AGENT_MOTION,
    ActionKind.PICK_UP: Mechanism.PICKUP,
    ActionKind.DROP: Mechanism.DROP,
    ActionKind.PUT_IN: Mechanism.CONTAINMENT,
    ActionKind.TAKE_OUT: Mechanism.CONTAINMENT,
    ActionKind.OPEN_CONTAINER: Mechanism.CONTAINER_ACCESS,
    ActionKind.CLOSE_CONTAINER: Mechanism.CONTAINER_ACCESS,
    ActionKind.MOVE_CONTAINER: Mechanism.CONTAINER_TRANSPORT,
}


_BOUNDARY_EVENT_REASON: Mapping[EventKind, BoundaryReason] = {
    EventKind.OBJECT_PICKED_UP: BoundaryReason.RELATION_CHANGE,
    EventKind.OBJECT_DROPPED: BoundaryReason.RELATION_CHANGE,
    EventKind.OBJECT_PLACED_IN_CONTAINER: BoundaryReason.RELATION_CHANGE,
    EventKind.OBJECT_REMOVED_FROM_CONTAINER: BoundaryReason.RELATION_CHANGE,
    EventKind.CONTAINER_OPENED: BoundaryReason.CONTAINER_STATE_CHANGE,
    EventKind.CONTAINER_CLOSED: BoundaryReason.CONTAINER_STATE_CHANGE,
    EventKind.CONTAINER_MOVED: BoundaryReason.TRANSPORT_COMPLETED,
    EventKind.ENTITY_BECAME_VISIBLE: BoundaryReason.VISIBILITY_CHANGE,
    EventKind.ENTITY_BECAME_HIDDEN: BoundaryReason.VISIBILITY_CHANGE,
}


def observe(state: WorldState, config: WorldConfig) -> Observation:
    """Return a causal, partial observation at ``state.step_index``.

    Visibility is local (Chebyshev radius) and an object in a closed container is
    hidden even when the container itself is visible.  No future state is used.
    """

    entities = {entity.entity_id: entity for entity in state.entities}
    visible: List[EntityObservation] = []
    for entity in sorted(state.entities, key=lambda item: item.entity_id):
        if entity.entity_id == state.held_object_id:
            is_visible = True
        elif entity.contained_in is not None:
            container = entities.get(entity.contained_in)
            is_visible = bool(
                container is not None
                and container.is_open
                and state.agent_position.chebyshev_distance(container.position)
                <= config.observation_radius
            )
        else:
            is_visible = (
                state.agent_position.chebyshev_distance(entity.position)
                <= config.observation_radius
            )

        if is_visible:
            visible.append(
                EntityObservation(
                    entity_id=entity.entity_id,
                    kind=entity.kind,
                    position=entity.position,
                    contained_in=entity.contained_in,
                    is_open=entity.is_open,
                    color=entity.color,
                    shape=entity.shape,
                    role=entity.role,
                )
            )

    return Observation(
        step_index=state.step_index,
        agent_position=state.agent_position,
        held_object_id=state.held_object_id,
        visible_entities=tuple(visible),
    )


def _require_target(action: Action) -> str:
    if action.target_id is None:
        raise InvalidActionError(f"{action.kind.value} requires target_id")
    return action.target_id


def _require_container_id(action: Action) -> str:
    if action.container_id is None:
        raise InvalidActionError(f"{action.kind.value} requires container_id")
    return action.container_id


def _require_destination(action: Action, config: WorldConfig) -> Position:
    if action.destination is None:
        raise InvalidActionError(f"{action.kind.value} requires destination")
    if not config.contains(action.destination):
        raise InvalidActionError(f"destination {action.destination} is out of bounds")
    return action.destination


def _require_entity(
    entities: Mapping[str, EntityState], entity_id: str, kind: EntityKind
) -> EntityState:
    try:
        entity = entities[entity_id]
    except KeyError as exc:
        raise InvalidActionError(f"unknown entity: {entity_id}") from exc
    if entity.kind is not kind:
        raise InvalidActionError(
            f"entity {entity_id} is {entity.kind.value}, expected {kind.value}"
        )
    return entity


def _require_cardinal_step(origin: Position, destination: Position) -> None:
    distance = abs(origin.x - destination.x) + abs(origin.y - destination.y)
    if distance != 1:
        raise InvalidActionError("movement must be exactly one cardinal grid step")


def _event_attributes(**values: object) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((key, str(value)) for key, value in values.items()))


def _apply_action(
    state: WorldState, action: Action, config: WorldConfig
) -> Tuple[WorldState, Tuple[Event, ...]]:
    """Apply mechanics without deriving observations, boundaries, or causality."""

    next_step = state.step_index + 1
    entities: MutableMapping[str, EntityState] = {
        entity.entity_id: entity for entity in state.entities
    }
    agent_position = state.agent_position
    held_object_id = state.held_object_id
    events: List[Event] = []

    def add_event(
        kind: EventKind, participants: Iterable[str], **attributes: object
    ) -> None:
        events.append(
            Event(
                event_id=f"event:{next_step}:{len(events)}",
                step_index=next_step,
                kind=kind,
                participants=tuple(participants),
                attributes=_event_attributes(**attributes),
            )
        )

    if action.kind is ActionKind.WAIT:
        pass

    elif action.kind is ActionKind.MOVE_AGENT:
        destination = _require_destination(action, config)
        _require_cardinal_step(agent_position, destination)
        old_position = agent_position
        agent_position = destination
        add_event(
            EventKind.AGENT_MOVED,
            ("agent",),
            source=old_position.as_tuple(),
            destination=destination.as_tuple(),
        )
        if held_object_id is not None:
            held = _require_entity(entities, held_object_id, EntityKind.OBJECT)
            entities[held_object_id] = replace(held, position=destination)
            add_event(
                EventKind.OBJECT_CARRIED,
                ("agent", held_object_id),
                source=old_position.as_tuple(),
                destination=destination.as_tuple(),
            )

    elif action.kind is ActionKind.PICK_UP:
        target_id = _require_target(action)
        target = _require_entity(entities, target_id, EntityKind.OBJECT)
        if held_object_id is not None:
            raise InvalidActionError("agent is already holding an object")
        if target.contained_in is not None:
            raise InvalidActionError("use take_out for an object in a container")
        if target.position != agent_position:
            raise InvalidActionError("agent and object must occupy the same cell")
        held_object_id = target_id
        add_event(EventKind.OBJECT_PICKED_UP, ("agent", target_id))

    elif action.kind is ActionKind.DROP:
        if held_object_id is None:
            raise InvalidActionError("agent is not holding an object")
        if action.target_id is not None and action.target_id != held_object_id:
            raise InvalidActionError("drop target does not match held object")
        target_id = held_object_id
        target = _require_entity(entities, target_id, EntityKind.OBJECT)
        entities[target_id] = replace(
            target, position=agent_position, contained_in=None
        )
        held_object_id = None
        add_event(
            EventKind.OBJECT_DROPPED,
            ("agent", target_id),
            position=agent_position.as_tuple(),
        )

    elif action.kind is ActionKind.PUT_IN:
        target_id = _require_target(action)
        container_id = _require_container_id(action)
        if held_object_id != target_id:
            raise InvalidActionError("put_in target must be the held object")
        target = _require_entity(entities, target_id, EntityKind.OBJECT)
        container = _require_entity(entities, container_id, EntityKind.CONTAINER)
        if not container.is_open:
            raise InvalidActionError("container must be open")
        if container.position != agent_position:
            raise InvalidActionError("agent and container must occupy the same cell")
        entities[target_id] = replace(
            target, position=container.position, contained_in=container_id
        )
        held_object_id = None
        add_event(
            EventKind.OBJECT_PLACED_IN_CONTAINER,
            ("agent", target_id, container_id),
        )

    elif action.kind is ActionKind.TAKE_OUT:
        target_id = _require_target(action)
        container_id = _require_container_id(action)
        if held_object_id is not None:
            raise InvalidActionError("agent is already holding an object")
        target = _require_entity(entities, target_id, EntityKind.OBJECT)
        container = _require_entity(entities, container_id, EntityKind.CONTAINER)
        if target.contained_in != container_id:
            raise InvalidActionError("object is not in the requested container")
        if not container.is_open:
            raise InvalidActionError("container must be open")
        if container.position != agent_position:
            raise InvalidActionError("agent and container must occupy the same cell")
        entities[target_id] = replace(
            target, position=agent_position, contained_in=None
        )
        held_object_id = target_id
        add_event(
            EventKind.OBJECT_REMOVED_FROM_CONTAINER,
            ("agent", target_id, container_id),
        )

    elif action.kind in (ActionKind.OPEN_CONTAINER, ActionKind.CLOSE_CONTAINER):
        target_id = _require_target(action)
        container = _require_entity(entities, target_id, EntityKind.CONTAINER)
        if container.position != agent_position:
            raise InvalidActionError("agent and container must occupy the same cell")
        should_open = action.kind is ActionKind.OPEN_CONTAINER
        if container.is_open == should_open:
            state_name = "open" if should_open else "closed"
            raise InvalidActionError(f"container is already {state_name}")
        entities[target_id] = replace(container, is_open=should_open)
        add_event(
            EventKind.CONTAINER_OPENED
            if should_open
            else EventKind.CONTAINER_CLOSED,
            ("agent", target_id),
        )

    elif action.kind is ActionKind.MOVE_CONTAINER:
        target_id = _require_target(action)
        destination = _require_destination(action, config)
        container = _require_entity(entities, target_id, EntityKind.CONTAINER)
        if container.position != agent_position:
            raise InvalidActionError("agent must stand at the container's source cell")
        _require_cardinal_step(container.position, destination)
        old_position = container.position
        entities[target_id] = replace(container, position=destination)
        add_event(
            EventKind.CONTAINER_MOVED,
            ("agent", target_id),
            source=old_position.as_tuple(),
            destination=destination.as_tuple(),
        )
        for entity_id, entity in tuple(entities.items()):
            if entity.contained_in == target_id:
                entities[entity_id] = replace(entity, position=destination)
                add_event(
                    EventKind.CONTAINED_OBJECT_MOVED,
                    (target_id, entity_id),
                    source=old_position.as_tuple(),
                    destination=destination.as_tuple(),
                )

    else:  # pragma: no cover - Enum exhaustiveness guard
        raise InvalidActionError(f"unsupported action: {action.kind}")

    return (
        WorldState(
            step_index=next_step,
            agent_position=agent_position,
            held_object_id=held_object_id,
            entities=tuple(sorted(entities.values(), key=lambda item: item.entity_id)),
        ),
        tuple(events),
    )


def _with_visibility_events(
    events: Tuple[Event, ...],
    before: Observation,
    after: Observation,
) -> Tuple[Event, ...]:
    result = list(events)
    before_ids = set(before.visible_entity_ids)
    after_ids = set(after.visible_entity_ids)
    for entity_id in sorted(after_ids - before_ids):
        result.append(
            Event(
                event_id=f"event:{after.step_index}:{len(result)}",
                step_index=after.step_index,
                kind=EventKind.ENTITY_BECAME_VISIBLE,
                participants=(entity_id,),
            )
        )
    for entity_id in sorted(before_ids - after_ids):
        result.append(
            Event(
                event_id=f"event:{after.step_index}:{len(result)}",
                step_index=after.step_index,
                kind=EventKind.ENTITY_BECAME_HIDDEN,
                participants=(entity_id,),
            )
        )
    return tuple(result)


def _boundary_from_events(
    step_index: int, events: Tuple[Event, ...], config: WorldConfig
) -> BoundaryLabel:
    triggering = tuple(event for event in events if event.kind in _BOUNDARY_EVENT_REASON)
    reasons = tuple(
        sorted(
            {_BOUNDARY_EVENT_REASON[event.kind] for event in triggering},
            key=lambda reason: reason.value,
        )
    )
    return BoundaryLabel(
        step_index=step_index,
        is_boundary=bool(reasons),
        reasons=reasons,
        event_ids=tuple(event.event_id for event in triggering),
        ontology_version=config.ontology_version,
    )


def _causal_state_values(state: WorldState) -> Dict[str, object]:
    values: Dict[str, object] = {
        "agent.position": state.agent_position.as_tuple(),
        "agent.held_object_id": state.held_object_id,
    }
    for entity in state.entities:
        prefix = f"entity.{entity.entity_id}"
        values[f"{prefix}.position"] = entity.position.as_tuple()
        values[f"{prefix}.contained_in"] = entity.contained_in
        if entity.kind is EntityKind.CONTAINER:
            values[f"{prefix}.is_open"] = entity.is_open
    return values


def derive_causal_edges(
    action: Action,
    factual_after: WorldState,
    counterfactual_after: WorldState,
) -> Tuple[CausalEdge, ...]:
    """Derive direct action effects from a deterministic do-intervention."""

    if action.kind is ActionKind.WAIT:
        return ()
    mechanism = _ACTION_MECHANISM[action.kind]
    factual = _causal_state_values(factual_after)
    counterfactual = _causal_state_values(counterfactual_after)
    source = f"action:{factual_after.step_index}:{action.kind.value}"
    edges = []
    for target in sorted(factual):
        if factual[target] != counterfactual[target]:
            edges.append(
                CausalEdge(
                    source=source,
                    target=target,
                    mechanism=mechanism,
                    evidence=InterventionEvidence(
                        method=CAUSAL_EDGE_METHOD,
                        intervention="do(action=wait)",
                        factual_value=factual[target],
                        counterfactual_value=counterfactual[target],
                    ),
                )
            )
    return tuple(edges)


def simulate_step(
    state: WorldState, action: Action, config: WorldConfig
) -> StepRecord:
    """Pure deterministic transition including a no-action counterfactual."""

    before_observation = observe(state, config)
    after, events = _apply_action(state, action, config)
    counterfactual_after, _ = _apply_action(state, Action.wait(), config)
    after_observation = observe(after, config)
    events = _with_visibility_events(events, before_observation, after_observation)
    return StepRecord(
        step_index=after.step_index,
        before=state,
        action=action,
        after=after,
        counterfactual_after=counterfactual_after,
        observation=after_observation,
        events=events,
        boundary=_boundary_from_events(after.step_index, events, config),
        causal_edges=derive_causal_edges(action, after, counterfactual_after),
    )


class SymbolicWorld:
    """Small stateful wrapper around the pure transition function."""

    def __init__(self, initial_state: WorldState, config: WorldConfig) -> None:
        self._initial_state = initial_state
        self._state = initial_state
        self.config = config

    @property
    def state(self) -> WorldState:
        return self._state

    def observation(self) -> Observation:
        return observe(self._state, self.config)

    def reset(self) -> Observation:
        self._state = self._initial_state
        return self.observation()

    def step(self, action: Action) -> StepRecord:
        record = simulate_step(self._state, action, self.config)
        self._state = record.after
        return record

    def run(self, actions: Iterable[Action]) -> Tuple[StepRecord, ...]:
        return tuple(self.step(action) for action in actions)
