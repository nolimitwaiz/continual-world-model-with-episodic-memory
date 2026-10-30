"""Immutable schemas for the symbolic CMWM benchmark.

The first prototype deliberately uses compact, JSON-friendly symbolic values.  No
tensor or rendering dependency belongs in these types.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


EVENT_ONTOLOGY_VERSION = "cmwm.events.v1"
CAUSAL_EDGE_METHOD = "single_step_do_action_wait_v1"
MECHANISM_SPLIT_POLICY = "held_out_mechanism_v1"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class EntityKind(str, Enum):
    OBJECT = "object"
    CONTAINER = "container"


class ActionKind(str, Enum):
    WAIT = "wait"
    MOVE_AGENT = "move_agent"
    PICK_UP = "pick_up"
    DROP = "drop"
    PUT_IN = "put_in"
    TAKE_OUT = "take_out"
    OPEN_CONTAINER = "open_container"
    CLOSE_CONTAINER = "close_container"
    MOVE_CONTAINER = "move_container"


class Mechanism(str, Enum):
    AGENT_MOTION = "agent_motion"
    PICKUP = "pickup"
    DROP = "drop"
    CONTAINMENT = "containment"
    CONTAINER_ACCESS = "container_access"
    CONTAINER_TRANSPORT = "container_transport"


class EventKind(str, Enum):
    AGENT_MOVED = "agent_moved"
    OBJECT_CARRIED = "object_carried"
    OBJECT_PICKED_UP = "object_picked_up"
    OBJECT_DROPPED = "object_dropped"
    OBJECT_PLACED_IN_CONTAINER = "object_placed_in_container"
    OBJECT_REMOVED_FROM_CONTAINER = "object_removed_from_container"
    CONTAINER_OPENED = "container_opened"
    CONTAINER_CLOSED = "container_closed"
    CONTAINER_MOVED = "container_moved"
    CONTAINED_OBJECT_MOVED = "contained_object_moved"
    ENTITY_BECAME_VISIBLE = "entity_became_visible"
    ENTITY_BECAME_HIDDEN = "entity_became_hidden"


class BoundaryReason(str, Enum):
    """Versioned semantic reasons for a ground-truth event boundary.

    Ordinary agent motion is intentionally not a semantic boundary.  A motion
    step can still become a boundary if it changes the visible entity set.
    """

    RELATION_CHANGE = "relation_change"
    CONTAINER_STATE_CHANGE = "container_state_change"
    TRANSPORT_COMPLETED = "transport_completed"
    VISIBILITY_CHANGE = "visibility_change"


class QueryKind(str, Enum):
    DELAYED_STATE = "delayed_state"
    OBJECT_LOCATION_HISTORY = "object_location_history"
    OCCLUSION_CONTINUITY = "occlusion_continuity"
    CAUSAL_PREDECESSOR = "causal_predecessor"


@dataclass(frozen=True, order=True)
class Position:
    x: int
    y: int

    def as_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def chebyshev_distance(self, other: "Position") -> int:
        return max(abs(self.x - other.x), abs(self.y - other.y))


@dataclass(frozen=True)
class WorldConfig:
    width: int = 6
    height: int = 4
    observation_radius: int = 1
    ontology_version: str = EVENT_ONTOLOGY_VERSION

    def __post_init__(self) -> None:
        if self.width < 5:
            raise ValueError("width must be at least 5")
        if self.height < 2:
            raise ValueError("height must be at least 2")
        if self.observation_radius < 0:
            raise ValueError("observation_radius must be non-negative")

    def contains(self, position: Position) -> bool:
        return 0 <= position.x < self.width and 0 <= position.y < self.height


@dataclass(frozen=True)
class EntityState:
    entity_id: str
    kind: EntityKind
    position: Position
    contained_in: Optional[str] = None
    is_open: bool = False
    color: str = "unspecified"
    shape: str = "unspecified"
    role: str = "unspecified"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    target_id: Optional[str] = None
    container_id: Optional[str] = None
    destination: Optional[Position] = None

    @classmethod
    def wait(cls) -> "Action":
        return cls(ActionKind.WAIT)


@dataclass(frozen=True)
class WorldState:
    step_index: int
    agent_position: Position
    held_object_id: Optional[str]
    entities: Tuple[EntityState, ...]

    def entity(self, entity_id: str) -> EntityState:
        for entity in self.entities:
            if entity.entity_id == entity_id:
                return entity
        raise KeyError(entity_id)


@dataclass(frozen=True)
class EntityObservation:
    entity_id: str
    kind: EntityKind
    position: Position
    contained_in: Optional[str]
    is_open: bool
    color: str = "unspecified"
    shape: str = "unspecified"
    role: str = "unspecified"


@dataclass(frozen=True)
class Observation:
    step_index: int
    agent_position: Position
    held_object_id: Optional[str]
    visible_entities: Tuple[EntityObservation, ...]

    @property
    def visible_entity_ids(self) -> Tuple[str, ...]:
        return tuple(entity.entity_id for entity in self.visible_entities)


@dataclass(frozen=True)
class Event:
    event_id: str
    step_index: int
    kind: EventKind
    participants: Tuple[str, ...]
    attributes: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class BoundaryLabel:
    step_index: int
    is_boundary: bool
    reasons: Tuple[BoundaryReason, ...]
    event_ids: Tuple[str, ...]
    ontology_version: str = EVENT_ONTOLOGY_VERSION


CausalValue = Optional[object]


@dataclass(frozen=True)
class InterventionEvidence:
    method: str
    intervention: str
    factual_value: CausalValue
    counterfactual_value: CausalValue


@dataclass(frozen=True)
class CausalEdge:
    """A direct action effect supported by an explicit counterfactual.

    The edge exists iff the target state variable differs between the factual
    transition and the same transition under ``do(action=WAIT)``.
    """

    source: str
    target: str
    mechanism: Mechanism
    evidence: InterventionEvidence


@dataclass(frozen=True)
class StepRecord:
    step_index: int
    before: WorldState
    action: Action
    after: WorldState
    counterfactual_after: WorldState
    observation: Observation
    events: Tuple[Event, ...]
    boundary: BoundaryLabel
    causal_edges: Tuple[CausalEdge, ...]


@dataclass(frozen=True)
class MechanismSplitMetadata:
    dataset_split: DatasetSplit
    policy_id: str
    episode_seed: int
    template_id: str
    training_mechanisms: Tuple[Mechanism, ...]
    held_out_mechanisms: Tuple[Mechanism, ...]
    episode_mechanisms: Tuple[Mechanism, ...]
    mechanism_signature: str
    held_out_family: Optional[str] = None


@dataclass(frozen=True)
class Episode:
    episode_id: str
    config: WorldConfig
    initial_state: WorldState
    initial_observation: Observation
    steps: Tuple[StepRecord, ...]
    split_metadata: MechanismSplitMetadata

    @property
    def boundary_steps(self) -> Tuple[int, ...]:
        return tuple(
            record.step_index for record in self.steps if record.boundary.is_boundary
        )

    @property
    def causal_edges(self) -> Tuple[CausalEdge, ...]:
        return tuple(edge for record in self.steps for edge in record.causal_edges)


@dataclass(frozen=True)
class StructuredQuery:
    """A non-linguistic query whose answer can be computed exactly."""

    query_id: str
    kind: QueryKind
    ask_step: int
    reference_step: int
    target_entity_id: Optional[str] = None
    target_field: Optional[str] = None
    causal_target: Optional[str] = None
    causal_value: CausalValue = None


@dataclass(frozen=True)
class QueryExample:
    query: StructuredQuery
    oracle_answer: object
