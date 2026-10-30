from __future__ import annotations

from cmwm.g0 import _segment_evidence
from cmwm.schemas import DatasetSplit, Mechanism, QueryKind
from cmwm.synthetic.audit import audit_g0
from cmwm.synthetic.causal_controls import causal_controls_for_dataset
from cmwm.synthetic.generator import (
    ENDPOINT_BALANCED_POLICY,
    SHORTCUT_RESISTANT_POLICY,
    SyntheticEpisodeGenerator,
)
from cmwm.synthetic.queries import answer_query, effective_query_lag, generate_queries
from cmwm.synthetic.validation import episode_checksum, validate_dataset, validate_episode


def test_generation_is_deterministic_for_seed_and_index() -> None:
    generator = SyntheticEpisodeGenerator(base_seed=101)
    left = generator.generate_episode(DatasetSplit.TRAIN, 3)
    right = generator.generate_episode(DatasetSplit.TRAIN, 3)

    assert episode_checksum(left) == episode_checksum(right)


def test_test_split_contains_two_distinct_held_out_families() -> None:
    generator = SyntheticEpisodeGenerator(base_seed=7)
    family_a = generator.generate_episode(DatasetSplit.TEST, 0)
    family_b = generator.generate_episode(DatasetSplit.TEST, 1)

    assert family_a.split_metadata.held_out_family == "HF-A"
    assert family_b.split_metadata.held_out_family == "HF-B"
    assert Mechanism.CONTAINER_TRANSPORT not in family_a.split_metadata.episode_mechanisms
    assert Mechanism.CONTAINER_TRANSPORT in family_b.split_metadata.episode_mechanisms


def test_causal_edges_have_counterfactual_support() -> None:
    episode = SyntheticEpisodeGenerator(base_seed=17).generate_episode(
        DatasetSplit.TEST, 1
    )

    assert episode.causal_edges
    assert all(
        edge.evidence.factual_value != edge.evidence.counterfactual_value
        for edge in episode.causal_edges
    )
    validate_episode(episode)


def test_small_cross_split_fixture_passes_dataset_validation() -> None:
    generator = SyntheticEpisodeGenerator(base_seed=29)
    episodes = (
        generator.generate_episode(DatasetSplit.TRAIN, 0),
        generator.generate_episode(DatasetSplit.VALIDATION, 0),
        generator.generate_episode(DatasetSplit.TEST, 0),
        generator.generate_episode(DatasetSplit.TEST, 1),
    )

    validate_dataset(episodes)


def test_structured_queries_have_exact_recomputable_answers() -> None:
    episode = SyntheticEpisodeGenerator(base_seed=31).generate_episode(
        DatasetSplit.TEST, 1
    )
    examples = generate_queries(episode)

    assert examples
    assert all(
        answer_query(episode, example.query) == example.oracle_answer
        for example in examples
    )


def test_causal_queries_do_not_disclose_the_source_record() -> None:
    episode = SyntheticEpisodeGenerator(base_seed=41).generate_episode(
        DatasetSplit.TEST, 1
    )
    causal = [
        example
        for example in generate_queries(episode)
        if example.query.kind is QueryKind.CAUSAL_PREDECESSOR
    ]

    assert causal
    for example in causal:
        source_step = next(
            record.step_index
            for record in episode.steps
            if any(edge.source == example.oracle_answer for edge in record.causal_edges)
        )
        assert example.query.reference_step == example.query.ask_step
        assert source_step != example.query.reference_step
        assert str(example.oracle_answer) not in example.query.query_id


def test_repaired_generator_fixture_passes_default_g0_contract() -> None:
    generator = SyntheticEpisodeGenerator(base_seed=1729)
    episodes = tuple(
        generator.generate_episode(split, index)
        for split in DatasetSplit
        for index in range(4)
    )
    queries = {episode.episode_id: generate_queries(episode) for episode in episodes}
    controls = causal_controls_for_dataset(episodes, queries)
    oracle_lengths, random_lengths = _segment_evidence(episodes, seed=99)
    report = audit_g0(
        episodes,
        queries,
        causal_candidates_by_query=controls,
        oracle_segment_lengths_by_episode=oracle_lengths,
        random_segment_lengths_by_episode=random_lengths,
    )

    assert report.passed, sorted({finding.code for finding in report.errors})


def test_e2_generator_constructs_every_task_at_exact_primary_lags() -> None:
    generator = SyntheticEpisodeGenerator(
        base_seed=1729,
        schedule_policy=ENDPOINT_BALANCED_POLICY,
    )
    for episode_index in range(6):
        expected_lag = 64 if episode_index < 3 else 128
        episode = generator.generate_episode(DatasetSplit.VALIDATION, episode_index)
        examples = generate_queries(episode)
        observed = {
            example.query.kind
            for example in examples
            if effective_query_lag(episode, example) == expected_lag
        }
        assert observed == set(QueryKind)
        assert episode.episode_id.startswith("validation-e2-")


def test_e2_anchor_does_not_disclose_causal_source_step() -> None:
    episode = SyntheticEpisodeGenerator(
        base_seed=41,
        schedule_policy=ENDPOINT_BALANCED_POLICY,
    ).generate_episode(DatasetSplit.VALIDATION, 3)
    causal = [
        example
        for example in generate_queries(episode)
        if example.query.kind is QueryKind.CAUSAL_PREDECESSOR
        and effective_query_lag(episode, example) == 128
    ]
    assert causal
    assert all(example.query.reference_step == example.query.ask_step for example in causal)


def test_e3_constructs_confusable_primary_cells_without_current_state_answers() -> None:
    generator = SyntheticEpisodeGenerator(
        base_seed=1729,
        schedule_policy=SHORTCUT_RESISTANT_POLICY,
    )
    for episode_index in range(24):
        expected_lag = 64 if episode_index < 12 else 128
        episode = generator.generate_episode(DatasetSplit.VALIDATION, episode_index)
        examples = generate_queries(episode)
        observed = {
            example.query.kind
            for example in examples
            if effective_query_lag(episode, example) == expected_lag
        }
        assert observed == set(QueryKind)
        assert episode.episode_id.startswith("validation-e3-")
        assert sum(
            entity.kind.value == "object"
            for entity in episode.initial_state.entities
        ) == 2
        assert sum(
            entity.kind.value == "container"
            for entity in episode.initial_state.entities
        ) == 2

        for example in examples:
            query = example.query
            if query.kind is QueryKind.CAUSAL_PREDECESSOR:
                continue
            current = episode.steps[query.ask_step - 1].after.entity(
                str(query.target_entity_id)
            )
            current_value = (
                current.position.as_tuple()
                if query.target_field == "position"
                else current.contained_in
            )
            assert example.oracle_answer != current_value
