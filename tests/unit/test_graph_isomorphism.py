from __future__ import annotations

from dataclasses import replace

from cmwm.schemas import DatasetSplit
from cmwm.synthetic.generator import SyntheticEpisodeGenerator
from cmwm.synthetic.graph_isomorphism import (
    LabeledDirectedGraph,
    LabeledEdge,
    are_label_preserving_isomorphic,
    causal_action_windows,
    episode_causal_graph,
    motif_fingerprints,
    relabel_graph,
    verified_isomorphism_groups,
    wl_fingerprint,
)


def _permutation(graph: LabeledDirectedGraph) -> dict[str, str]:
    identifiers = [node.node_id for node in graph.nodes]
    return {
        node_id: f"renamed:{index:03d}"
        for index, node_id in enumerate(reversed(identifiers))
    }


def test_wl_and_exact_match_are_invariant_to_node_id_permutation() -> None:
    episode = SyntheticEpisodeGenerator(base_seed=101).generate_episode(
        DatasetSplit.TRAIN, 1
    )
    graph = episode_causal_graph(episode)
    renamed = relabel_graph(graph, _permutation(graph))

    assert wl_fingerprint(graph).digest == wl_fingerprint(renamed).digest
    assert are_label_preserving_isomorphic(graph, renamed)


def test_causal_edge_change_breaks_exact_isomorphism() -> None:
    episode = SyntheticEpisodeGenerator(base_seed=103).generate_episode(
        DatasetSplit.TEST, 1
    )
    graph = episode_causal_graph(episode)
    first = graph.edges[0]
    changed = replace(first, label="causal:adversarial_change")
    perturbed = LabeledDirectedGraph(
        schema_version=graph.schema_version,
        nodes=graph.nodes,
        edges=tuple(sorted((changed, *graph.edges[1:]))),
    )

    assert not are_label_preserving_isomorphic(graph, perturbed)


def test_same_template_groups_across_randomized_surface_identities() -> None:
    generator = SyntheticEpisodeGenerator(base_seed=107)
    episodes = tuple(
        generator.generate_episode(DatasetSplit.TRAIN, index) for index in range(9)
    )
    graphs = tuple(episode_causal_graph(episode) for episode in episodes)
    groups = verified_isomorphism_groups(graphs)
    grouped_templates = [
        {episodes[index].split_metadata.template_id for index in group}
        for group in groups
    ]

    assert len(groups) == 3
    assert all(len(templates) == 1 for templates in grouped_templates)
    assert sorted(len(group) for group in groups) == [3, 3, 3]


def test_action_windows_expose_local_motifs_without_entity_ids() -> None:
    episode = SyntheticEpisodeGenerator(base_seed=109).generate_episode(
        DatasetSplit.TEST, 1
    )
    graph = episode_causal_graph(episode)
    windows = causal_action_windows(graph, action_window=3)
    fingerprints = motif_fingerprints(graph, action_window=3)
    entity_ids = {entity.entity_id for entity in episode.initial_state.entities}

    assert windows
    assert len(windows) == len(fingerprints)
    assert all(len(fingerprint) == 64 for fingerprint in fingerprints)
    assert all(
        entity_id not in node.label
        for window in windows
        for node in window.nodes
        for entity_id in entity_ids
    )
