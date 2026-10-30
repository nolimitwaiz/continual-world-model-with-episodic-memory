from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import product

from cmwm.schemas import (
    CAUSAL_EDGE_METHOD,
    MECHANISM_SPLIT_POLICY,
    Action,
    ActionKind,
    BoundaryLabel,
    CausalEdge,
    DatasetSplit,
    EntityKind,
    EntityObservation,
    EntityState,
    Episode,
    InterventionEvidence,
    Mechanism,
    MechanismSplitMetadata,
    Observation,
    Position,
    QueryExample,
    QueryKind,
    StepRecord,
    StructuredQuery,
    WorldConfig,
    WorldState,
)
from cmwm.synthetic.audit import (
    CausalCandidateControl,
    G0AuditPolicy,
    _majority_label_without,
    audit_g0,
    summarize_query_balance,
)
from cmwm.synthetic.manifest import (
    assert_manifest_sealed,
    build_dataset_manifest,
    manifest_identity,
    seal_manifest,
    verify_manifest_seal,
)


def _state(step: int, object_x: int) -> WorldState:
    return WorldState(
        step_index=step,
        agent_position=Position(0, 0),
        held_object_id=None,
        entities=(
            EntityState(
                entity_id="object_0",
                kind=EntityKind.OBJECT,
                position=Position(object_x, 0),
            ),
        ),
    )


def _observation(step: int, object_x: int) -> Observation:
    return Observation(
        step_index=step,
        agent_position=Position(0, 0),
        held_object_id=None,
        visible_entities=(
            EntityObservation(
                entity_id="object_0",
                kind=EntityKind.OBJECT,
                position=Position(object_x, 0),
                contained_in=None,
                is_open=False,
            ),
        ),
    )


def _episode(
    episode_id: str,
    split: DatasetSplit,
    seed: int,
    template_id: str,
    object_x: int,
    *,
    family: str | None = None,
    mechanism: Mechanism | None = None,
    length: int = 4,
) -> Episode:
    steps = []
    before = _state(0, object_x)
    for step in range(1, length + 1):
        after = _state(step, object_x)
        causal_edges = ()
        action = Action.wait()
        if mechanism is not None and step == length:
            action = Action(ActionKind.PICK_UP, target_id="object_0")
            causal_edges = (
                CausalEdge(
                    source=f"action:{length}:pickup",
                    target="entity.object_0.position",
                    mechanism=mechanism,
                    evidence=InterventionEvidence(
                        method=CAUSAL_EDGE_METHOD,
                        intervention="do(action=wait)",
                        factual_value=(object_x, 0),
                        counterfactual_value=(object_x + 1, 0),
                    ),
                ),
            )
        steps.append(
            StepRecord(
                step_index=step,
                before=before,
                action=action,
                after=after,
                counterfactual_after=after,
                observation=_observation(step, object_x),
                events=(),
                boundary=BoundaryLabel(step, False, (), ()),
                causal_edges=causal_edges,
            )
        )
        before = after

    mechanisms = (mechanism,) if mechanism is not None else ()
    return Episode(
        episode_id=episode_id,
        config=WorldConfig(),
        initial_state=_state(0, object_x),
        initial_observation=_observation(0, object_x),
        steps=tuple(steps),
        split_metadata=MechanismSplitMetadata(
            dataset_split=split,
            policy_id=MECHANISM_SPLIT_POLICY,
            episode_seed=seed,
            template_id=template_id,
            training_mechanisms=(Mechanism.PICKUP,),
            held_out_mechanisms=(Mechanism.CONTAINER_TRANSPORT,),
            episode_mechanisms=mechanisms,
            mechanism_signature="+".join(item.value for item in mechanisms),
            held_out_family=family,
        ),
    )


def _state_query(
    episode: Episode,
    query_id: str,
    *,
    kind: QueryKind = QueryKind.DELAYED_STATE,
    reference_step: int = 0,
) -> QueryExample:
    query = StructuredQuery(
        query_id=query_id,
        kind=kind,
        ask_step=len(episode.steps),
        reference_step=reference_step,
        target_entity_id="object_0",
        target_field="position",
    )
    return QueryExample(query, episode.initial_state.entity("object_0").position.as_tuple())


def _causal_query(episode: Episode, query_id: str) -> QueryExample:
    edge = episode.steps[-1].causal_edges[0]
    query = StructuredQuery(
        query_id=query_id,
        kind=QueryKind.CAUSAL_PREDECESSOR,
        ask_step=len(episode.steps),
        reference_step=len(episode.steps),
        causal_target="entity.object_0.position",
        causal_value=edge.evidence.factual_value,
    )
    return QueryExample(query, edge.source)


def _fixture_policy(**changes: object) -> G0AuditPolicy:
    base = G0AuditPolicy(
        required_query_kinds=(),
        required_lags=(),
        required_test_families=(),
        min_label_examples=20,
        min_shortcut_examples=50,
        require_random_matched_evidence=False,
        min_training_templates=0,
        require_entity_id_diversity=False,
        require_event_timing_diversity=False,
        require_hf_a_appearance_role_fields=False,
    )
    return replace(base, **changes)


def test_manifest_is_order_independent_and_detects_query_changes() -> None:
    first = _episode("train-a", DatasetSplit.TRAIN, 11, "known-a", 1)
    second = _episode("train-b", DatasetSplit.TRAIN, 12, "known-b", 2)
    first_query = _state_query(first, "train-a:q")
    second_query = _state_query(second, "train-b:q")

    left = build_dataset_manifest(
        (first, second),
        {first.episode_id: (first_query,), second.episode_id: (second_query,)},
    )
    right = build_dataset_manifest(
        (second, first),
        {second.episode_id: (second_query,), first.episode_id: (first_query,)},
    )
    seal = seal_manifest(left)

    assert manifest_identity(left) == manifest_identity(right)
    assert verify_manifest_seal(right, seal)
    assert_manifest_sealed(right, seal)

    changed_query = replace(first_query, oracle_answer=(4, 0))
    changed = build_dataset_manifest(
        (first, second),
        {first.episode_id: (changed_query,), second.episode_id: (second_query,)},
    )
    assert not verify_manifest_seal(changed, seal)

    with_sidecar = build_dataset_manifest(
        (first, second),
        {first.episode_id: (first_query,), second.episode_id: (second_query,)},
        audit_sidecars={"candidate_controls": {"q": ("candidate-a",)}},
    )
    changed_sidecar = build_dataset_manifest(
        (first, second),
        {first.episode_id: (first_query,), second.episode_id: (second_query,)},
        audit_sidecars={"candidate_controls": {"q": ("candidate-b",)}},
    )
    assert manifest_identity(with_sidecar) != manifest_identity(changed_sidecar)


def test_audit_detects_seed_template_trajectory_and_mechanism_leakage() -> None:
    train = _episode(
        "train-a",
        DatasetSplit.TRAIN,
        21,
        "shared-template",
        1,
        mechanism=Mechanism.CONTAINER_TRANSPORT,
    )
    test = _episode(
        "test-a",
        DatasetSplit.TEST,
        21,
        "shared-template",
        1,
        family="HF-A",
        mechanism=Mechanism.CONTAINER_TRANSPORT,
    )

    report = audit_g0((train, test), policy=_fixture_policy())
    codes = {finding.code for finding in report.findings}

    assert "seed_cross_split_leakage" in codes
    assert "test_template_leakage" in codes
    assert "trajectory_cross_split_leakage" in codes
    assert "held_out_mechanism_leakage" in codes


def test_query_balance_summarizes_lag_task_and_held_out_family() -> None:
    family_a = _episode(
        "test-a", DatasetSplit.TEST, 31, "held-a", 1, family="HF-A"
    )
    family_b = _episode(
        "test-b",
        DatasetSplit.TEST,
        32,
        "held-b",
        2,
        family="HF-B",
        mechanism=Mechanism.CONTAINER_TRANSPORT,
    )
    queries = {
        family_a.episode_id: (_state_query(family_a, "test-a:q", reference_step=0),),
        family_b.episode_id: (
            _state_query(
                family_b,
                "test-b:q",
                kind=QueryKind.OBJECT_LOCATION_HISTORY,
                reference_step=2,
            ),
        ),
    }

    summary = summarize_query_balance((family_a, family_b), queries)

    assert summary.total_queries == 2
    assert {(item.name, item.count) for item in summary.by_family} == {
        ("HF-A", 1),
        ("HF-B", 1),
    }
    assert {(item.lag, item.count) for item in summary.by_lag} == {(2, 1), (4, 1)}
    assert {item.task for item in summary.cells} == {
        QueryKind.DELAYED_STATE.value,
        QueryKind.OBJECT_LOCATION_HISTORY.value,
    }


def test_storage_pressure_confirms_primary_reference_is_outside_and_evicted() -> None:
    episode = _episode(
        "train-long",
        DatasetSplit.TRAIN,
        35,
        "known-long",
        1,
        length=80,
    )
    example = _state_query(episode, "train-long:q", reference_step=16)

    report = audit_g0(
        (episode,),
        {episode.episode_id: (example,)},
        policy=_fixture_policy(),
    )

    assert report.passed
    assert report.storage_pressure.capacity_records == 64
    assert report.storage_pressure.episodes_with_eviction == 1
    assert report.storage_pressure.observation_records_evicted == 17
    assert report.storage_pressure.primary_queries == 1
    assert report.storage_pressure.primary_references_outside_context == 1
    assert report.storage_pressure.primary_references_evicted == 1


def test_random_segments_must_match_full_oracle_length_distribution() -> None:
    episode = _episode("train-segments", DatasetSplit.TRAIN, 36, "known", 1)
    matching = audit_g0(
        (episode,),
        oracle_segment_lengths_by_episode={episode.episode_id: (1, 3)},
        random_segment_lengths_by_episode={episode.episode_id: (3, 1)},
        policy=_fixture_policy(require_random_matched_evidence=True),
    )
    assert matching.segment_length_match.distributions_match
    assert "random_matched_length_distribution_mismatch" not in {
        finding.code for finding in matching.findings
    }

    mismatched = audit_g0(
        (episode,),
        oracle_segment_lengths_by_episode={episode.episode_id: (1, 3)},
        random_segment_lengths_by_episode={episode.episode_id: (2, 2)},
        policy=_fixture_policy(require_random_matched_evidence=True),
    )
    assert "random_matched_length_distribution_mismatch" in {
        finding.code for finding in mismatched.findings
    }


def test_audit_marks_current_diversity_claims_as_scaffold_only() -> None:
    episodes = (
        _episode("train-one", DatasetSplit.TRAIN, 37, "known", 1),
        _episode("train-two", DatasetSplit.TRAIN, 38, "known", 2),
        _episode(
            "test-hf-a",
            DatasetSplit.TEST,
            39,
            "held-a",
            1,
            family="HF-A",
        ),
    )
    report = audit_g0(
        episodes,
        policy=_fixture_policy(
            min_training_templates=3,
            require_entity_id_diversity=True,
            require_event_timing_diversity=True,
            require_hf_a_appearance_role_fields=True,
        ),
    )
    codes = {finding.code for finding in report.findings}

    assert "fixed_entity_id_shortcut" in codes
    assert "insufficient_training_template_diversity" in codes
    assert "deterministic_event_timing_shortcut" in codes
    assert "hf_a_appearance_role_scaffold_only" in codes
    assert "collision_mechanics_not_evidenced" in codes


def test_audit_flags_degenerate_labels_and_metadata_shortcuts() -> None:
    degenerate_episodes = tuple(
        _episode(f"deg-{index}", DatasetSplit.TRAIN, 40 + index, "mixed", 1)
        for index in range(3)
    )
    degenerate_queries = {
        episode.episode_id: (_state_query(episode, f"{episode.episode_id}:q"),)
        for episode in degenerate_episodes
    }
    degenerate = audit_g0(
        degenerate_episodes,
        degenerate_queries,
        policy=_fixture_policy(min_label_examples=3),
    )
    assert "degenerate_label" in {finding.code for finding in degenerate.findings}

    shortcut_episodes = tuple(
        _episode(
            f"shortcut-{index}",
            DatasetSplit.TRAIN,
            50 + index,
            "left" if index < 3 else "right",
            1 if index < 3 else 2,
        )
        for index in range(6)
    )
    shortcut_queries = {
        episode.episode_id: (_state_query(episode, f"{episode.episode_id}:q"),)
        for episode in shortcut_episodes
    }
    shortcut = audit_g0(
        shortcut_episodes,
        shortcut_queries,
        policy=_fixture_policy(
            min_shortcut_examples=4,
            min_shortcut_coverage=0.5,
            shortcut_accuracy_threshold=0.9,
            shortcut_uplift_threshold=0.4,
        ),
    )

    assert any(
        diagnostic.feature == "template_id" and diagnostic.flagged
        for diagnostic in shortcut.shortcut_diagnostics
    )
    assert "single_field_shortcut" in {
        finding.code for finding in shortcut.findings
    }


def test_causal_audit_flags_reference_and_unmatched_distractor_shortcuts() -> None:
    episode = _episode(
        "train-causal",
        DatasetSplit.TRAIN,
        70,
        "causal-template",
        1,
        mechanism=Mechanism.PICKUP,
    )
    example = _causal_query(episode, "train-causal:q")
    candidates = (
        CausalCandidateControl(
            source=example.oracle_answer,
            recency=0,
            entity_overlap=1.0,
            target_field="position",
            semantic_similarity=0.9,
        ),
        CausalCandidateControl(
            source="action:1:drop",
            recency=3,
            entity_overlap=0.0,
            target_field="contained_in",
            semantic_similarity=0.1,
        ),
    )

    report = audit_g0(
        (episode,),
        {episode.episode_id: (example,)},
        causal_candidates_by_query={example.query.query_id: candidates},
        policy=_fixture_policy(),
    )
    codes = {finding.code for finding in report.findings}

    assert "causal_reference_step_points_to_answer" in codes
    assert "causal_source_encodes_reference_step" in codes
    assert "causal_recency_unmatched" in codes
    assert "causal_entity_overlap_unmatched" in codes
    assert "causal_target_field_unmatched" in codes
    assert "causal_semantic_similarity_unmatched" in codes


def test_optimized_leave_episode_out_majority_matches_brute_force() -> None:
    labels = ("a", "b", "c")
    for raw_counts in product(range(4), repeat=len(labels)):
        counts = Counter(
            {label: count for label, count in zip(labels, raw_counts) if count}
        )
        if not counts:
            continue
        ranking = tuple(
            sorted(counts, key=lambda label: (-counts[label], label))
        )
        exclusion_ranges = [range(counts.get(label, 0) + 1) for label in labels]
        for raw_excluded in product(*exclusion_ranges):
            excluded = Counter(
                {
                    label: count
                    for label, count in zip(labels, raw_excluded)
                    if count
                }
            )
            remaining = Counter(
                {
                    label: counts[label] - excluded.get(label, 0)
                    for label in counts
                    if counts[label] - excluded.get(label, 0) > 0
                }
            )
            expected = (
                min(remaining, key=lambda label: (-remaining[label], label))
                if remaining
                else None
            )
            assert _majority_label_without(counts, excluded, ranking) == expected
