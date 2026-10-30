#!/usr/bin/env python3
"""Run the cluster-only GI-0 structural invariance and motif-coverage audit."""

from __future__ import annotations

import argparse
import json
import os
import platform
from collections import Counter, defaultdict
from pathlib import Path

from cmwm.infrastructure.artifacts import atomic_write_json
from cmwm.schemas import DatasetSplit
from cmwm.synthetic.generator import SyntheticEpisodeGenerator
from cmwm.synthetic.graph_isomorphism import (
    are_label_preserving_isomorphic,
    causal_action_windows,
    episode_causal_graph,
    relabel_graph,
    verified_isomorphism_groups,
    wl_fingerprint,
)


def _permutation(graph) -> dict[str, str]:
    ids = [node.node_id for node in graph.nodes]
    return {node_id: f"permuted:{index:04d}" for index, node_id in enumerate(reversed(ids))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=4242)
    parser.add_argument("--train-episodes", type=int, default=30)
    parser.add_argument("--validation-episodes", type=int, default=15)
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--action-window", type=int, default=3)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("GI-0 audit refuses to run outside a Slurm allocation")
    if min(args.train_episodes, args.validation_episodes, args.test_episodes) < 1:
        raise SystemExit("episode counts must be positive")

    generator = SyntheticEpisodeGenerator(base_seed=args.base_seed)
    dataset = generator.generate_dataset(
        args.train_episodes, args.validation_episodes, args.test_episodes
    )
    episodes = tuple(
        episode
        for split in (DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST)
        for episode in dataset[split]
    )
    graphs = tuple(episode_causal_graph(episode) for episode in episodes)
    groups = verified_isomorphism_groups(graphs)
    fingerprints = tuple(wl_fingerprint(graph).digest for graph in graphs)

    template_to_fingerprints: dict[str, set[str]] = defaultdict(set)
    fingerprint_to_templates: dict[str, set[str]] = defaultdict(set)
    for episode, fingerprint in zip(episodes, fingerprints):
        template = episode.split_metadata.template_id
        template_to_fingerprints[template].add(fingerprint)
        fingerprint_to_templates[fingerprint].add(template)

    permutation_checks = []
    for template in sorted(template_to_fingerprints):
        index = next(
            i for i, episode in enumerate(episodes)
            if episode.split_metadata.template_id == template
        )
        graph = graphs[index]
        renamed = relabel_graph(graph, _permutation(graph))
        permutation_checks.append({
            "template_id": template,
            "wl_equal": wl_fingerprint(graph).digest == wl_fingerprint(renamed).digest,
            "exact_isomorphic": are_label_preserving_isomorphic(graph, renamed),
        })

    train_episode_graphs = [
        episode_causal_graph(episode) for episode in dataset[DatasetSplit.TRAIN]
    ]
    train_motif_graphs = tuple(
        motif
        for graph in train_episode_graphs
        for motif in causal_action_windows(graph, action_window=args.action_window)
    )
    train_motif_index: dict[str, list] = defaultdict(list)
    for motif in train_motif_graphs:
        train_motif_index[wl_fingerprint(motif).digest].append(motif)
    train_motif_groups = verified_isomorphism_groups(train_motif_graphs)
    split_motif_coverage = {}
    for split in DatasetSplit:
        split_graphs = [episode_causal_graph(episode) for episode in dataset[split]]
        per_episode_wl = []
        per_episode_exact = []
        wl_candidate_count = 0
        exact_match_count = 0
        for graph in split_graphs:
            motifs = causal_action_windows(graph, action_window=args.action_window)
            wl_matches = 0
            exact_matches = 0
            for motif in motifs:
                candidates = train_motif_index.get(wl_fingerprint(motif).digest, ())
                if candidates:
                    wl_matches += 1
                    wl_candidate_count += 1
                if any(
                    are_label_preserving_isomorphic(motif, candidate)
                    for candidate in candidates
                ):
                    exact_matches += 1
                    exact_match_count += 1
            denominator = len(motifs)
            per_episode_wl.append(wl_matches / denominator if denominator else 0.0)
            per_episode_exact.append(exact_matches / denominator if denominator else 0.0)
        split_motif_coverage[split.value] = {
            "episodes": len(per_episode_exact),
            "mean_fraction_with_wl_candidate": sum(per_episode_wl) / len(per_episode_wl),
            "mean_fraction_exact_verified_in_train": sum(per_episode_exact) / len(per_episode_exact),
            "minimum_exact_verified_fraction": min(per_episode_exact),
            "maximum_exact_verified_fraction": max(per_episode_exact),
            "wl_candidate_motifs": wl_candidate_count,
            "exact_verified_motifs": exact_match_count,
            "wl_candidate_false_positives": wl_candidate_count - exact_match_count,
        }

    group_rows = []
    for group_id, group in enumerate(groups):
        group_rows.append({
            "group_id": group_id,
            "episodes": [episodes[index].episode_id for index in group],
            "templates": sorted({episodes[index].split_metadata.template_id for index in group}),
            "splits": dict(sorted(Counter(
                episodes[index].split_metadata.dataset_split.value for index in group
            ).items())),
        })
    passed = (
        all(len(values) == 1 for values in template_to_fingerprints.values())
        and all(len(values) == 1 for values in fingerprint_to_templates.values())
        and all(row["wl_equal"] and row["exact_isomorphic"] for row in permutation_checks)
        and all(len(row["templates"]) == 1 for row in group_rows)
    )
    result = {
        "artifact_type": "cmwm_graph_isomorphism_gi0_audit",
        "schema_version": "cmwm.gi0-audit.v2",
        "claim_eligible": False,
        "training_performed": False,
        "passed": passed,
        "interpretation": (
            "Engineering/exploratory structural audit only. Whole-graph exact matches "
            "identify known templates; local motif coverage estimates whether held-out "
            "compositions reuse training structure. No memory advantage is tested."
        ),
        "limitations": [
            "1-WL fingerprints are incomplete and are used only as an index before exact verification.",
            "Generator templates may make structural classification easier than natural event graphs.",
            "Exact whole-graph matching cannot generalize to genuinely novel topology; motif coverage is diagnostic only.",
            "No FilmFest or Sherlock causal graph is used in GI-0.",
        ],
        "parameters": {
            "base_seed": args.base_seed,
            "train_episodes": args.train_episodes,
            "validation_episodes": args.validation_episodes,
            "test_episodes": args.test_episodes,
            "wl_iterations": 4,
            "action_window": args.action_window,
        },
        "episodes": len(episodes),
        "verified_isomorphism_groups": group_rows,
        "template_to_wl_fingerprints": {
            template: sorted(values) for template, values in sorted(template_to_fingerprints.items())
        },
        "wl_fingerprint_collisions_across_templates": {
            fingerprint: sorted(templates)
            for fingerprint, templates in sorted(fingerprint_to_templates.items())
            if len(templates) > 1
        },
        "permutation_checks": permutation_checks,
        "train_motif_wl_vocabulary_size": len(train_motif_index),
        "train_motif_exact_equivalence_classes": len(train_motif_groups),
        "split_motif_coverage": split_motif_coverage,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
