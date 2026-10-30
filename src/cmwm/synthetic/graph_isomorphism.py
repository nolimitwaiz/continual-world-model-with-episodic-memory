"""Permutation-invariant causal-event graph utilities for the GI-0 audit.

GI-0 is an exploratory structural-transfer audit, not a Prototype 1 memory
condition. Weisfeiler--Leman (WL) fingerprints are used only as a fast index;
label-preserving exact matching verifies candidate equivalence because 1-WL is
not a complete graph-isomorphism test.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from cmwm.schemas import EntityKind, Episode


GRAPH_SCHEMA_VERSION = "cmwm.causal-event-graph.v1"
WL_SCHEMA_VERSION = "cmwm.wl-fingerprint.v1"


class GraphValidationError(ValueError):
    """Raised when a labeled graph violates the GI-0 contract."""


class GraphIsomorphismLimitError(RuntimeError):
    """Raised when exact verification exceeds its declared search budget."""


@dataclass(frozen=True, order=True)
class LabeledNode:
    node_id: str
    label: str


@dataclass(frozen=True, order=True)
class LabeledEdge:
    source: str
    target: str
    label: str


@dataclass(frozen=True)
class LabeledDirectedGraph:
    schema_version: str
    nodes: tuple[LabeledNode, ...]
    edges: tuple[LabeledEdge, ...]

    def __post_init__(self) -> None:
        if self.schema_version != GRAPH_SCHEMA_VERSION:
            raise GraphValidationError("unknown causal-event graph schema")
        node_ids = [node.node_id for node in self.nodes]
        if not self.nodes or len(node_ids) != len(set(node_ids)):
            raise GraphValidationError("graph nodes must be non-empty and unique")
        known = set(node_ids)
        if any(edge.source not in known or edge.target not in known for edge in self.edges):
            raise GraphValidationError("graph edge references an unknown node")
        if len(self.edges) != len(set(self.edges)):
            raise GraphValidationError("duplicate labeled edges are forbidden")

    @property
    def node_labels(self) -> dict[str, str]:
        return {node.node_id: node.label for node in self.nodes}


@dataclass(frozen=True)
class WLFingerprint:
    schema_version: str
    iterations: int
    digest: str
    node_count: int
    edge_count: int


def _sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _target_label(target: str, entity_kinds: Mapping[str, EntityKind]) -> str:
    if target.startswith("agent."):
        return f"state:{target}"
    parts = target.split(".")
    if len(parts) != 3 or parts[0] != "entity":
        raise GraphValidationError(f"unknown causal target schema: {target}")
    entity_id, field = parts[1], parts[2]
    try:
        kind = entity_kinds[entity_id]
    except KeyError as exc:
        raise GraphValidationError(f"causal target references unknown entity: {target}") from exc
    return f"state:{kind.value}:{field}"


def episode_causal_graph(episode: Episode) -> LabeledDirectedGraph:
    """Build a surface-identity-invariant temporal causal-effect graph.

    Action nodes retain action type. Each intervention-supported state effect
    becomes its own node labeled by entity kind and affected field, not by the
    randomized entity ID or value. Consecutive causal actions receive a
    directed temporal edge. This preserves causal topology and order while
    making entity renaming irrelevant.
    """

    entity_kinds = {
        entity.entity_id: entity.kind for entity in episode.initial_state.entities
    }
    nodes: list[LabeledNode] = []
    edges: list[LabeledEdge] = []
    previous_action: str | None = None
    action_ordinal = 0
    for record in episode.steps:
        if not record.causal_edges:
            continue
        action_id = f"action:{action_ordinal:03d}"
        nodes.append(LabeledNode(action_id, f"action:{record.action.kind.value}"))
        if previous_action is not None:
            edges.append(LabeledEdge(previous_action, action_id, "temporal:next"))
        for effect_ordinal, causal_edge in enumerate(record.causal_edges):
            effect_id = f"effect:{action_ordinal:03d}:{effect_ordinal:03d}"
            nodes.append(
                LabeledNode(effect_id, _target_label(causal_edge.target, entity_kinds))
            )
            edges.append(
                LabeledEdge(
                    action_id,
                    effect_id,
                    f"causal:{causal_edge.mechanism.value}",
                )
            )
        previous_action = action_id
        action_ordinal += 1
    if not nodes:
        raise GraphValidationError(f"episode has no intervention-supported edges: {episode.episode_id}")
    return LabeledDirectedGraph(
        schema_version=GRAPH_SCHEMA_VERSION,
        nodes=tuple(sorted(nodes)),
        edges=tuple(sorted(edges)),
    )


def wl_fingerprint(
    graph: LabeledDirectedGraph, *, iterations: int = 4
) -> WLFingerprint:
    """Return a deterministic directed, edge-labeled 1-WL fingerprint."""

    if iterations < 1:
        raise ValueError("WL iterations must be positive")
    labels = graph.node_labels
    incoming: dict[str, list[tuple[str, str]]] = defaultdict(list)
    outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in graph.edges:
        outgoing[edge.source].append((edge.label, edge.target))
        incoming[edge.target].append((edge.label, edge.source))
    colors = {node_id: _sha256(["node", label]) for node_id, label in labels.items()}
    history: list[list[tuple[str, int]]] = []
    history.append(sorted(Counter(colors.values()).items()))
    for _ in range(iterations):
        refined = {}
        for node_id in sorted(labels):
            in_colors = sorted(
                (edge_label, colors[source])
                for edge_label, source in incoming[node_id]
            )
            out_colors = sorted(
                (edge_label, colors[target])
                for edge_label, target in outgoing[node_id]
            )
            refined[node_id] = _sha256(
                ["refine", colors[node_id], ["in", in_colors], ["out", out_colors]]
            )
        colors = refined
        history.append(sorted(Counter(colors.values()).items()))
    edge_colors = sorted(
        (colors[edge.source], edge.label, colors[edge.target]) for edge in graph.edges
    )
    digest = _sha256(
        {
            "schema": WL_SCHEMA_VERSION,
            "iterations": iterations,
            "history": history,
            "final_edges": edge_colors,
        }
    )
    return WLFingerprint(
        schema_version=WL_SCHEMA_VERSION,
        iterations=iterations,
        digest=digest,
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
    )


def relabel_graph(
    graph: LabeledDirectedGraph, mapping: Mapping[str, str]
) -> LabeledDirectedGraph:
    """Return the same labeled graph under a complete node-ID permutation."""

    node_ids = {node.node_id for node in graph.nodes}
    if set(mapping) != node_ids or len(set(mapping.values())) != len(node_ids):
        raise GraphValidationError("relabeling must be a bijection over every node")
    return LabeledDirectedGraph(
        schema_version=graph.schema_version,
        nodes=tuple(
            sorted(LabeledNode(mapping[node.node_id], node.label) for node in graph.nodes)
        ),
        edges=tuple(
            sorted(
                LabeledEdge(mapping[edge.source], mapping[edge.target], edge.label)
                for edge in graph.edges
            )
        ),
    )


def _node_signatures(graph: LabeledDirectedGraph) -> dict[str, tuple]:
    incoming: dict[str, Counter[str]] = defaultdict(Counter)
    outgoing: dict[str, Counter[str]] = defaultdict(Counter)
    for edge in graph.edges:
        outgoing[edge.source][edge.label] += 1
        incoming[edge.target][edge.label] += 1
    return {
        node.node_id: (
            node.label,
            tuple(sorted(incoming[node.node_id].items())),
            tuple(sorted(outgoing[node.node_id].items())),
        )
        for node in graph.nodes
    }


def are_label_preserving_isomorphic(
    left: LabeledDirectedGraph,
    right: LabeledDirectedGraph,
    *,
    max_search_states: int = 250_000,
) -> bool:
    """Verify directed node/edge-label isomorphism with bounded backtracking."""

    if max_search_states < 1:
        raise ValueError("max_search_states must be positive")
    if len(left.nodes) != len(right.nodes) or len(left.edges) != len(right.edges):
        return False
    left_signatures = _node_signatures(left)
    right_signatures = _node_signatures(right)
    if Counter(left_signatures.values()) != Counter(right_signatures.values()):
        return False
    left_edge_counts = Counter((edge.source, edge.target, edge.label) for edge in left.edges)
    right_edge_counts = Counter((edge.source, edge.target, edge.label) for edge in right.edges)
    candidates = {
        node_id: tuple(
            sorted(
                other_id
                for other_id, signature in right_signatures.items()
                if signature == left_signatures[node_id]
            )
        )
        for node_id in left_signatures
    }
    mapping: dict[str, str] = {}
    used: set[str] = set()
    searched = 0

    def compatible(left_id: str, right_id: str) -> bool:
        if left_edge_counts[(left_id, left_id, "temporal:next")] != right_edge_counts[
            (right_id, right_id, "temporal:next")
        ]:
            return False
        for mapped_left, mapped_right in mapping.items():
            labels = {
                edge.label
                for edge in left.edges
                if (edge.source, edge.target) in {
                    (left_id, mapped_left),
                    (mapped_left, left_id),
                }
            } | {
                edge.label
                for edge in right.edges
                if (edge.source, edge.target) in {
                    (right_id, mapped_right),
                    (mapped_right, right_id),
                }
            }
            for label in labels:
                if left_edge_counts[(left_id, mapped_left, label)] != right_edge_counts[
                    (right_id, mapped_right, label)
                ]:
                    return False
                if left_edge_counts[(mapped_left, left_id, label)] != right_edge_counts[
                    (mapped_right, right_id, label)
                ]:
                    return False
        return True

    def search() -> bool:
        nonlocal searched
        searched += 1
        if searched > max_search_states:
            raise GraphIsomorphismLimitError(
                f"exact isomorphism exceeded {max_search_states} search states"
            )
        if len(mapping) == len(left.nodes):
            mapped_edges = Counter(
                (mapping[edge.source], mapping[edge.target], edge.label)
                for edge in left.edges
            )
            return mapped_edges == right_edge_counts
        remaining = [node_id for node_id in left_signatures if node_id not in mapping]
        left_id = min(
            remaining,
            key=lambda node_id: (
                sum(candidate not in used for candidate in candidates[node_id]),
                node_id,
            ),
        )
        for right_id in candidates[left_id]:
            if right_id in used or not compatible(left_id, right_id):
                continue
            mapping[left_id] = right_id
            used.add(right_id)
            if search():
                return True
            used.remove(right_id)
            del mapping[left_id]
        return False

    return search()


def induced_subgraph(
    graph: LabeledDirectedGraph, node_ids: Iterable[str]
) -> LabeledDirectedGraph:
    selected = set(node_ids)
    if not selected or not selected.issubset(graph.node_labels):
        raise GraphValidationError("induced subgraph requires known non-empty nodes")
    return LabeledDirectedGraph(
        schema_version=graph.schema_version,
        nodes=tuple(sorted(node for node in graph.nodes if node.node_id in selected)),
        edges=tuple(
            sorted(
                edge
                for edge in graph.edges
                if edge.source in selected and edge.target in selected
            )
        ),
    )


def causal_action_windows(
    graph: LabeledDirectedGraph, *, action_window: int = 3
) -> tuple[LabeledDirectedGraph, ...]:
    """Extract consecutive-action motifs with their direct effect nodes."""

    if action_window < 1:
        raise ValueError("action_window must be positive")
    action_ids = sorted(
        node.node_id for node in graph.nodes if node.label.startswith("action:")
    )
    outgoing_effects: dict[str, set[str]] = defaultdict(set)
    labels = graph.node_labels
    for edge in graph.edges:
        if edge.label.startswith("causal:") and labels[edge.target].startswith("state:"):
            outgoing_effects[edge.source].add(edge.target)
    windows = []
    for start in range(0, len(action_ids) - action_window + 1):
        actions = action_ids[start : start + action_window]
        selected = set(actions)
        for action_id in actions:
            selected.update(outgoing_effects[action_id])
        windows.append(induced_subgraph(graph, selected))
    return tuple(windows)


def motif_fingerprints(
    graph: LabeledDirectedGraph,
    *,
    action_window: int = 3,
    iterations: int = 4,
) -> tuple[str, ...]:
    return tuple(
        wl_fingerprint(window, iterations=iterations).digest
        for window in causal_action_windows(graph, action_window=action_window)
    )


def verified_isomorphism_groups(
    graphs: Sequence[LabeledDirectedGraph], *, iterations: int = 4
) -> tuple[tuple[int, ...], ...]:
    """Group graphs using WL indexing followed by exact verification."""

    buckets: dict[str, list[int]] = defaultdict(list)
    for index, graph in enumerate(graphs):
        buckets[wl_fingerprint(graph, iterations=iterations).digest].append(index)
    groups: list[list[int]] = []
    for digest in sorted(buckets):
        for index in buckets[digest]:
            for group in groups:
                if group[0] in buckets[digest] and are_label_preserving_isomorphic(
                    graphs[group[0]], graphs[index]
                ):
                    group.append(index)
                    break
            else:
                groups.append([index])
    return tuple(tuple(group) for group in groups)
