"""Seeded episode generation with an explicit mechanism-holdout policy."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple, Union

from cmwm.schemas import (
    MECHANISM_SPLIT_POLICY,
    Action,
    ActionKind,
    DatasetSplit,
    EntityKind,
    EntityState,
    Episode,
    Mechanism,
    MechanismSplitMetadata,
    Position,
    WorldConfig,
    WorldState,
)
from cmwm.synthetic.world import SymbolicWorld, observe


TRAINING_MECHANISMS: Tuple[Mechanism, ...] = tuple(
    sorted(
        (
            Mechanism.AGENT_MOTION,
            Mechanism.PICKUP,
            Mechanism.DROP,
            Mechanism.CONTAINMENT,
            Mechanism.CONTAINER_ACCESS,
        ),
        key=lambda mechanism: mechanism.value,
    )
)
HELD_OUT_MECHANISMS: Tuple[Mechanism, ...] = (
    Mechanism.CONTAINER_TRANSPORT,
)


_SPLIT_SEED_OFFSET = {
    DatasetSplit.TRAIN: 0,
    DatasetSplit.VALIDATION: 1_000_003,
    DatasetSplit.TEST: 2_000_003,
}

RANDOM_SCHEDULE_POLICY = "random_v1"
ENDPOINT_BALANCED_POLICY = "exact_task_lag_v2"
SHORTCUT_RESISTANT_POLICY = "confusable_events_v3"
_SCHEDULE_POLICIES = {
    RANDOM_SCHEDULE_POLICY,
    ENDPOINT_BALANCED_POLICY,
    SHORTCUT_RESISTANT_POLICY,
}


def _cardinal_path(start: Position, destination: Position) -> Tuple[Action, ...]:
    """Create a deterministic x-then-y cardinal path."""

    actions: List[Action] = []
    current = start
    while current.x != destination.x:
        direction = 1 if destination.x > current.x else -1
        current = Position(current.x + direction, current.y)
        actions.append(Action(ActionKind.MOVE_AGENT, destination=current))
    while current.y != destination.y:
        direction = 1 if destination.y > current.y else -1
        current = Position(current.x, current.y + direction)
        actions.append(Action(ActionKind.MOVE_AGENT, destination=current))
    return tuple(actions)


def _spread_actions(
    actions: Tuple[Action, ...], total_steps: int, rng: random.Random
) -> Tuple[Action, ...]:
    """Spread semantic actions over a seeded, order-preserving random schedule."""

    if len(actions) > total_steps:
        raise ValueError("semantic action sequence exceeds episode horizon")
    if not actions:
        return (Action.wait(),) * total_steps
    slots = sorted(rng.sample(range(total_steps), k=len(actions)))
    scheduled = [Action.wait() for _ in range(total_steps)]
    for slot, action in zip(slots, actions):
        scheduled[slot] = action
    return tuple(scheduled)


def _spread_actions_with_anchor(
    actions: Tuple[Action, ...],
    total_steps: int,
    *,
    anchor_index: int,
    anchor_step: int,
    rng: random.Random,
) -> Tuple[Action, ...]:
    """Spread actions while placing one semantic action at an exact 1-based step."""

    if not 0 <= anchor_index < len(actions):
        raise ValueError("anchor_index is outside the semantic action sequence")
    if not 1 <= anchor_step <= total_steps:
        raise ValueError("anchor_step is outside the episode horizon")
    anchor_slot = anchor_step - 1
    before_count = anchor_index
    after_count = len(actions) - anchor_index - 1
    if before_count > anchor_slot or after_count > total_steps - anchor_slot - 1:
        raise ValueError("semantic actions do not fit around the requested anchor")
    slots = sorted(rng.sample(range(anchor_slot), k=before_count))
    slots.append(anchor_slot)
    slots.extend(
        sorted(
            rng.sample(
                range(anchor_slot + 1, total_steps),
                k=after_count,
            )
        )
    )
    scheduled = [Action.wait() for _ in range(total_steps)]
    for slot, action in zip(slots, actions):
        scheduled[slot] = action
    return tuple(scheduled)


class SyntheticEpisodeGenerator:
    """Generate deterministic symbolic episodes from isolated local RNGs.

    ``CONTAINER_TRANSPORT`` never occurs in train/validation episodes and occurs
    in every test episode.  This policy tests a truly held-out mechanism rather
    than relying on a random frame split.
    """

    def __init__(
        self,
        base_seed: int = 0,
        config: WorldConfig | None = None,
        episode_steps: int = 160,
        schedule_policy: str = RANDOM_SCHEDULE_POLICY,
    ) -> None:
        self.base_seed = int(base_seed)
        self.config = config or WorldConfig()
        if episode_steps <= 0:
            raise ValueError("episode_steps must be positive")
        self.episode_steps = episode_steps
        if schedule_policy not in _SCHEDULE_POLICIES:
            raise ValueError(f"unknown schedule policy: {schedule_policy}")
        self.schedule_policy = schedule_policy

    def _episode_seed(self, split: DatasetSplit, episode_index: int) -> int:
        if episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        return self.base_seed + _SPLIT_SEED_OFFSET[split] + episode_index * 9_973

    def _initial_state(
        self, rng: random.Random, held_out_family: str | None
    ) -> Tuple[WorldState, Position, Position, Position, str, str]:
        row = rng.randrange(self.config.height)
        left_to_right = bool(rng.randrange(2))
        if left_to_right:
            agent = Position(0, row)
            obj = Position(1, row)
            container = Position(self.config.width - 2, row)
            transport_destination = Position(self.config.width - 1, row)
        else:
            agent = Position(self.config.width - 1, row)
            obj = Position(self.config.width - 2, row)
            container = Position(1, row)
            transport_destination = Position(0, row)

        object_id = f"object_{rng.getrandbits(32):08x}"
        container_id = f"container_{rng.getrandbits(32):08x}"
        if held_out_family == "HF-A":
            object_color = rng.choice(("cyan", "purple"))
            object_shape = rng.choice(("hexagon", "star"))
            object_role = "tracked_item"
            container_color = rng.choice(("teal", "gold"))
            container_shape = rng.choice(("urn", "capsule"))
            container_role = "occluder"
        else:
            object_color = rng.choice(("red", "blue", "green"))
            object_shape = rng.choice(("circle", "square", "triangle"))
            object_role = "payload"
            container_color = rng.choice(("gray", "brown", "white"))
            container_shape = rng.choice(("box", "basket", "chest"))
            container_role = "storage"

        state = WorldState(
            step_index=0,
            agent_position=agent,
            held_object_id=None,
            entities=(
                EntityState(
                    entity_id=container_id,
                    kind=EntityKind.CONTAINER,
                    position=container,
                    is_open=False,
                    color=container_color,
                    shape=container_shape,
                    role=container_role,
                ),
                EntityState(
                    entity_id=object_id,
                    kind=EntityKind.OBJECT,
                    position=obj,
                    color=object_color,
                    shape=object_shape,
                    role=object_role,
                ),
            ),
        )
        return (
            state,
            obj,
            container,
            transport_destination,
            object_id,
            container_id,
        )

    def _e3_initial_state(
        self, rng: random.Random
    ) -> tuple[
        WorldState,
        tuple[Position, Position],
        tuple[Position, Position],
        tuple[str, str],
        tuple[str, str],
    ]:
        """Create two interchangeable objects and two confusable containers.

        E2 had one object and one container, so a task/template label could stand
        in for an episode-specific binding.  E3 randomizes entity identities,
        target roles, orientation, and object-to-container assignment while
        staying inside the unchanged eight-slot model budget.
        """

        rows = (0, self.config.height - 1)
        left_to_right = bool(rng.randrange(2))
        object_x = 1 if left_to_right else self.config.width - 2
        container_x = self.config.width - 2 if left_to_right else 1
        agent_x = 0 if left_to_right else self.config.width - 1
        object_positions = tuple(Position(object_x, row) for row in rows)
        container_positions = tuple(Position(container_x, row) for row in rows)
        object_ids = tuple(
            f"object_{rng.getrandbits(32):08x}" for _ in range(2)
        )
        container_ids = tuple(
            f"container_{rng.getrandbits(32):08x}" for _ in range(2)
        )
        object_appearance = (
            ("red", "circle", "payload"),
            ("blue", "square", "payload"),
        )
        container_appearance = (
            ("gray", "box", "storage"),
            ("brown", "basket", "storage"),
        )
        entities = []
        for entity_id, position, appearance in zip(
            object_ids, object_positions, object_appearance
        ):
            entities.append(
                EntityState(
                    entity_id=entity_id,
                    kind=EntityKind.OBJECT,
                    position=position,
                    color=appearance[0],
                    shape=appearance[1],
                    role=appearance[2],
                )
            )
        for entity_id, position, appearance in zip(
            container_ids, container_positions, container_appearance
        ):
            entities.append(
                EntityState(
                    entity_id=entity_id,
                    kind=EntityKind.CONTAINER,
                    position=position,
                    is_open=False,
                    color=appearance[0],
                    shape=appearance[1],
                    role=appearance[2],
                )
            )
        state = WorldState(
            step_index=0,
            agent_position=Position(agent_x, rows[rng.randrange(2)]),
            held_object_id=None,
            entities=tuple(sorted(entities, key=lambda entity: entity.entity_id)),
        )
        return (
            state,
            object_positions,  # type: ignore[return-value]
            container_positions,  # type: ignore[return-value]
            object_ids,  # type: ignore[return-value]
            container_ids,  # type: ignore[return-value]
        )

    def _e3_confusable_actions(
        self,
        *,
        initial_state: WorldState,
        object_positions: tuple[Position, Position],
        container_positions: tuple[Position, Position],
        object_ids: tuple[str, str],
        container_ids: tuple[str, str],
        target_index: int,
        crossed_assignment: bool,
        route_variant: str,
        rng: random.Random,
    ) -> tuple[tuple[Action, ...], int]:
        """Build two matched containment episodes around one exact target anchor."""

        distractor_index = 1 - target_index
        assigned_container = {
            index: index ^ int(crossed_assignment) for index in range(2)
        }
        order = (distractor_index, target_index)
        current = initial_state.agent_position
        before: list[Action] = []
        if route_variant not in ("direct", "post_detour_y", "post_detour_x"):
            raise ValueError(f"unknown E3 route variant: {route_variant}")
        for object_index in order:
            object_id = object_ids[object_index]
            object_position = object_positions[object_index]
            container_index = assigned_container[object_index]
            container_id = container_ids[container_index]
            container_position = container_positions[container_index]
            before.extend(_cardinal_path(current, object_position))
            before.append(Action(ActionKind.PICK_UP, target_id=object_id))
            before.extend(_cardinal_path(object_position, container_position))
            before.extend(
                (
                    Action(ActionKind.OPEN_CONTAINER, target_id=container_id),
                    Action(
                        ActionKind.PUT_IN,
                        target_id=object_id,
                        container_id=container_id,
                    ),
                )
            )
            current = container_position
            if object_index == distractor_index:
                before.append(Action(ActionKind.CLOSE_CONTAINER, target_id=container_id))

        target_container_index = assigned_container[target_index]
        target_container_id = container_ids[target_container_index]
        target_container_position = container_positions[target_container_index]
        target_id = object_ids[target_index]
        anchor_index = len(before)
        actions = before + [
            Action(ActionKind.CLOSE_CONTAINER, target_id=target_container_id)
        ]

        # Change both objects again after the queried event.  Their final state
        # cannot answer the historical containment/location/occlusion query.
        final_candidates = tuple(
            Position(x, y)
            for x in range(self.config.width)
            for y in range(self.config.height)
            if Position(x, y) not in container_positions
        )
        target_destination = rng.choice(
            tuple(
                position
                for position in final_candidates
                if position != target_container_position
            )
        )
        actions.extend(
            (
                Action(ActionKind.OPEN_CONTAINER, target_id=target_container_id),
                Action(
                    ActionKind.TAKE_OUT,
                    target_id=target_id,
                    container_id=target_container_id,
                ),
            )
        )
        actions.extend(_cardinal_path(target_container_position, target_destination))
        actions.append(Action(ActionKind.DROP, target_id=target_id))
        current = target_destination
        if route_variant in ("post_detour_y", "post_detour_x"):
            if route_variant == "post_detour_y":
                detour = Position(current.x, (current.y + 1) % self.config.height)
            else:
                detour = Position((current.x + 1) % self.config.width, current.y)
            actions.extend(_cardinal_path(current, detour))
            actions.extend(_cardinal_path(detour, current))

        distractor_container_index = assigned_container[distractor_index]
        distractor_container_id = container_ids[distractor_container_index]
        distractor_container_position = container_positions[distractor_container_index]
        distractor_id = object_ids[distractor_index]
        distractor_destination = rng.choice(
            tuple(
                position
                for position in final_candidates
                if position != distractor_container_position
                and position != target_destination
            )
        )
        actions.extend(_cardinal_path(current, distractor_container_position))
        actions.extend(
            (
                Action(ActionKind.OPEN_CONTAINER, target_id=distractor_container_id),
                Action(
                    ActionKind.TAKE_OUT,
                    target_id=distractor_id,
                    container_id=distractor_container_id,
                ),
            )
        )
        actions.extend(
            _cardinal_path(distractor_container_position, distractor_destination)
        )
        actions.append(Action(ActionKind.DROP, target_id=distractor_id))
        return tuple(actions), anchor_index

    @staticmethod
    def _pickup_prefix(
        agent: Position, object_position: Position, object_id: str
    ) -> Tuple[Action, ...]:
        return _cardinal_path(agent, object_position) + (
            Action(ActionKind.PICK_UP, target_id=object_id),
        )

    def _known_actions(
        self,
        template_id: str,
        initial_state: WorldState,
        object_position: Position,
        container_position: Position,
        object_id: str,
        container_id: str,
    ) -> Tuple[Action, ...]:
        prefix = self._pickup_prefix(
            initial_state.agent_position, object_position, object_id
        )
        if template_id == "known_pick_drop_v1":
            direction = 1 if container_position.x > object_position.x else -1
            drop_position = Position(object_position.x + direction, object_position.y)
            return prefix + _cardinal_path(object_position, drop_position) + (
                Action(ActionKind.DROP, target_id=object_id),
            )

        if template_id == "known_relocation_return_v1":
            return (
                prefix
                + _cardinal_path(object_position, container_position)
                + (
                    Action(ActionKind.DROP, target_id=object_id),
                    Action(ActionKind.PICK_UP, target_id=object_id),
                )
                + _cardinal_path(container_position, object_position)
                + (Action(ActionKind.DROP, target_id=object_id),)
            )

        if template_id != "known_containment_cycle_v1":
            raise ValueError(f"unknown template: {template_id}")
        return (
            prefix
            + _cardinal_path(object_position, container_position)
            + (
                Action(ActionKind.OPEN_CONTAINER, target_id=container_id),
                Action(
                    ActionKind.PUT_IN,
                    target_id=object_id,
                    container_id=container_id,
                ),
                Action(ActionKind.CLOSE_CONTAINER, target_id=container_id),
                Action(ActionKind.OPEN_CONTAINER, target_id=container_id),
                Action(
                    ActionKind.TAKE_OUT,
                    target_id=object_id,
                    container_id=container_id,
                ),
                Action(ActionKind.DROP, target_id=object_id),
            )
        )

    def _endpoint_controlled_actions(
        self,
        template_id: str,
        initial_state: WorldState,
        object_position: Position,
        container_position: Position,
        object_id: str,
        container_id: str,
    ) -> tuple[Tuple[Action, ...], int]:
        """Build a containment-and-return trajectory with one hidden-state anchor."""

        prefix = self._pickup_prefix(
            initial_state.agent_position, object_position, object_id
        )
        if template_id == "endpoint_containment_direct_v2":
            approach = _cardinal_path(object_position, container_position)
        elif template_id == "endpoint_containment_drop_repick_v2":
            direction = 1 if container_position.x > object_position.x else -1
            waypoint = Position(object_position.x + direction, object_position.y)
            approach = (
                _cardinal_path(object_position, waypoint)
                + (
                    Action(ActionKind.DROP, target_id=object_id),
                    Action(ActionKind.PICK_UP, target_id=object_id),
                )
                + _cardinal_path(waypoint, container_position)
            )
        elif template_id == "endpoint_containment_detour_v2":
            detour = Position(
                object_position.x,
                (object_position.y + 1) % self.config.height,
            )
            approach = (
                _cardinal_path(object_position, detour)
                + _cardinal_path(detour, object_position)
                + _cardinal_path(object_position, container_position)
            )
        else:
            raise ValueError(f"unknown endpoint-controlled template: {template_id}")

        before_anchor = (
            prefix
            + approach
            + (
                Action(ActionKind.OPEN_CONTAINER, target_id=container_id),
                Action(
                    ActionKind.PUT_IN,
                    target_id=object_id,
                    container_id=container_id,
                ),
            )
        )
        anchor_index = len(before_anchor)
        actions = (
            before_anchor
            + (Action(ActionKind.CLOSE_CONTAINER, target_id=container_id),)
            + (
                Action(ActionKind.OPEN_CONTAINER, target_id=container_id),
                Action(
                    ActionKind.TAKE_OUT,
                    target_id=object_id,
                    container_id=container_id,
                ),
            )
            + _cardinal_path(container_position, object_position)
            + (Action(ActionKind.DROP, target_id=object_id),)
        )
        return actions, anchor_index

    def _held_out_actions(
        self,
        initial_state: WorldState,
        object_position: Position,
        container_position: Position,
        transport_destination: Position,
        object_id: str,
        container_id: str,
    ) -> Tuple[Action, ...]:
        return (
            self._pickup_prefix(initial_state.agent_position, object_position, object_id)
            + _cardinal_path(object_position, container_position)
            + (
                Action(ActionKind.OPEN_CONTAINER, target_id=container_id),
                Action(
                    ActionKind.PUT_IN,
                    target_id=object_id,
                    container_id=container_id,
                ),
                Action(ActionKind.CLOSE_CONTAINER, target_id=container_id),
                Action(
                    ActionKind.MOVE_CONTAINER,
                    target_id=container_id,
                    destination=transport_destination,
                ),
                Action(
                    ActionKind.MOVE_AGENT,
                    destination=transport_destination,
                ),
                Action(ActionKind.OPEN_CONTAINER, target_id=container_id),
                Action(
                    ActionKind.TAKE_OUT,
                    target_id=object_id,
                    container_id=container_id,
                ),
                Action(ActionKind.DROP, target_id=object_id),
            )
        )

    def _held_out_persistence_actions(
        self,
        initial_state: WorldState,
        object_position: Position,
        container_position: Position,
        object_id: str,
        container_id: str,
    ) -> Tuple[Action, ...]:
        """Hold the object out of view while the agent traverses away and back."""

        return (
            self._pickup_prefix(initial_state.agent_position, object_position, object_id)
            + _cardinal_path(object_position, container_position)
            + (
                Action(ActionKind.OPEN_CONTAINER, target_id=container_id),
                Action(
                    ActionKind.PUT_IN,
                    target_id=object_id,
                    container_id=container_id,
                ),
                Action(ActionKind.CLOSE_CONTAINER, target_id=container_id),
            )
            + _cardinal_path(container_position, object_position)
            + _cardinal_path(object_position, container_position)
            + (
                Action(ActionKind.OPEN_CONTAINER, target_id=container_id),
                Action(
                    ActionKind.TAKE_OUT,
                    target_id=object_id,
                    container_id=container_id,
                ),
                Action(ActionKind.DROP, target_id=object_id),
            )
        )

    def generate_episode(
        self,
        split: Union[DatasetSplit, str],
        episode_index: int = 0,
    ) -> Episode:
        split = DatasetSplit(split)
        episode_seed = self._episode_seed(split, episode_index)
        rng = random.Random(episode_seed)
        held_out_family = (
            "HF-A"
            if split is DatasetSplit.TEST and episode_index % 2 == 0
            else "HF-B" if split is DatasetSplit.TEST else None
        )
        if self.schedule_policy == SHORTCUT_RESISTANT_POLICY and held_out_family is None:
            (
                initial_state,
                object_positions,
                container_positions,
                object_ids,
                container_ids,
            ) = self._e3_initial_state(rng)
            object_position = object_positions[0]
            container_position = container_positions[0]
            transport_destination = container_positions[0]
            object_id = object_ids[0]
            container_id = container_ids[0]
        else:
            (
                initial_state,
                object_position,
                container_position,
                transport_destination,
                object_id,
                container_id,
            ) = self._initial_state(rng, held_out_family)

        anchor_index: int | None = None
        if self.schedule_policy == SHORTCUT_RESISTANT_POLICY and held_out_family is None:
            target_index = episode_index % 2
            route_variant = ("direct", "post_detour_y", "post_detour_x")[
                (episode_index // 2) % 3
            ]
            crossed_assignment = bool((episode_index // 6) % 2)
            template_id = f"confusable_{route_variant}_v3"
            actions, anchor_index = self._e3_confusable_actions(
                initial_state=initial_state,
                object_positions=object_positions,
                container_positions=container_positions,
                object_ids=object_ids,
                container_ids=container_ids,
                target_index=target_index,
                crossed_assignment=crossed_assignment,
                route_variant=route_variant,
                rng=rng,
            )
        elif self.schedule_policy == ENDPOINT_BALANCED_POLICY and held_out_family is None:
            template_id = (
                "endpoint_containment_direct_v2",
                "endpoint_containment_drop_repick_v2",
                "endpoint_containment_detour_v2",
            )[episode_index % 3]
            actions, anchor_index = self._endpoint_controlled_actions(
                template_id,
                initial_state,
                object_position,
                container_position,
                object_id,
                container_id,
            )
        elif held_out_family == "HF-A":
            template_id = "held_out_persistence_composition_v1"
            actions = self._held_out_persistence_actions(
                initial_state,
                object_position,
                container_position,
                object_id,
                container_id,
            )
        elif held_out_family == "HF-B":
            template_id = "held_out_container_transport_v1"
            actions = self._held_out_actions(
                initial_state,
                object_position,
                container_position,
                transport_destination,
                object_id,
                container_id,
            )
        else:
            template_id = (
                "known_pick_drop_v1",
                "known_containment_cycle_v1",
                "known_relocation_return_v1",
            )[episode_index % 3]
            actions = self._known_actions(
                template_id,
                initial_state,
                object_position,
                container_position,
                object_id,
                container_id,
            )

        if len(actions) > self.episode_steps:
            raise ValueError(
                f"episode_steps={self.episode_steps} is shorter than template "
                f"{template_id} ({len(actions)} actions)"
            )
        if anchor_index is None:
            actions = _spread_actions(actions, self.episode_steps, rng)
        else:
            if self.schedule_policy == SHORTCUT_RESISTANT_POLICY:
                target_lag = (64, 128)[(episode_index // 12) % 2]
            else:
                target_lag = (64, 128)[(episode_index // 3) % 2]
            anchor_step = self.episode_steps - target_lag
            actions = _spread_actions_with_anchor(
                actions,
                self.episode_steps,
                anchor_index=anchor_index,
                anchor_step=anchor_step,
                rng=rng,
            )
        world = SymbolicWorld(initial_state, self.config)
        steps = world.run(actions)
        episode_mechanisms = tuple(
            sorted(
                {
                    edge.mechanism
                    for record in steps
                    for edge in record.causal_edges
                },
                key=lambda mechanism: mechanism.value,
            )
        )
        mechanism_signature = "+".join(
            mechanism.value for mechanism in episode_mechanisms
        )
        metadata = MechanismSplitMetadata(
            dataset_split=split,
            policy_id=MECHANISM_SPLIT_POLICY,
            episode_seed=episode_seed,
            template_id=template_id,
            training_mechanisms=TRAINING_MECHANISMS,
            held_out_mechanisms=HELD_OUT_MECHANISMS,
            episode_mechanisms=episode_mechanisms,
            mechanism_signature=mechanism_signature,
            held_out_family=held_out_family,
        )
        return Episode(
            episode_id=(
                f"{split.value}-e2-{episode_index:05d}-{episode_seed}"
                if self.schedule_policy == ENDPOINT_BALANCED_POLICY
                else f"{split.value}-e3-{episode_index:05d}-{episode_seed}"
                if self.schedule_policy == SHORTCUT_RESISTANT_POLICY
                else f"{split.value}-{episode_index:05d}-{episode_seed}"
            ),
            config=self.config,
            initial_state=initial_state,
            initial_observation=observe(initial_state, self.config),
            steps=steps,
            split_metadata=metadata,
        )

    def generate_dataset(
        self,
        train_episodes: int,
        validation_episodes: int,
        test_episodes: int,
    ) -> Dict[DatasetSplit, Tuple[Episode, ...]]:
        counts = {
            DatasetSplit.TRAIN: train_episodes,
            DatasetSplit.VALIDATION: validation_episodes,
            DatasetSplit.TEST: test_episodes,
        }
        if any(count < 0 for count in counts.values()):
            raise ValueError("episode counts must be non-negative")
        return {
            split: tuple(self.generate_episode(split, index) for index in range(count))
            for split, count in counts.items()
        }
