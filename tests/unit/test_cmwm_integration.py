from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from cmwm.data import AnswerVocabulary, tensorize_examples
from cmwm.models.cmwm import CMWM, MemoryCondition
from cmwm.schemas import DatasetSplit, QueryKind
from cmwm.synthetic.generator import SyntheticEpisodeGenerator
from cmwm.synthetic.queries import generate_queries


def _episode_query_pairs():
    generator = SyntheticEpisodeGenerator(base_seed=83, episode_steps=40)
    episodes = [
        generator.generate_episode(DatasetSplit.TRAIN, 0),
        generator.generate_episode(DatasetSplit.TRAIN, 1),
    ]
    all_queries = [generate_queries(episode, lags=(16, 32)) for episode in episodes]
    state_example = next(
        example
        for example in all_queries[0]
        if example.query.kind is not QueryKind.CAUSAL_PREDECESSOR
    )
    causal_example = next(
        example
        for example in all_queries[1]
        if example.query.kind is QueryKind.CAUSAL_PREDECESSOR
    )
    return [(episodes[0], state_example), (episodes[1], causal_example)]


def _model(vocabulary: AnswerVocabulary) -> CMWM:
    return CMWM(
        max_entities=8,
        grid_width=6,
        grid_height=4,
        context_steps=8,
        answer_vocabulary=vocabulary,
        fixed_chunk_steps=5,
    )


def test_tensorization_has_stable_safe_targets() -> None:
    batch, vocabulary = tensorize_examples(_episode_query_pairs(), max_entities=8)

    assert batch.sequence.step_mask.shape == (2, 40)
    assert batch.sequence.entity_positions.shape == (2, 40, 8, 2)
    assert batch.answer_targets[0] >= 0
    assert batch.answer_targets[1] == -100
    assert batch.causal_source_steps[0] == -1
    assert batch.causal_source_steps[1] > 0
    assert vocabulary.size == 6 * 4 + 8 + 1 + 2

    repeated, _ = tensorize_examples(_episode_query_pairs(), max_entities=8)
    assert torch.equal(
        batch.query.target_entity_slot, repeated.query.target_entity_slot
    )
    assert torch.equal(batch.sequence.entity_mask, repeated.sequence.entity_mask)


@pytest.mark.parametrize("condition", list(MemoryCondition))
def test_integrated_model_executes_every_declared_memory_condition(
    condition: MemoryCondition,
) -> None:
    batch, vocabulary = tensorize_examples(_episode_query_pairs(), max_entities=8)
    model = _model(vocabulary)
    output = model(batch, condition=condition, random_seed=19)
    loss = model.loss(output, batch, condition=condition)

    assert output.answer_logits.shape == (2, vocabulary.size)
    assert output.candidate_logits.shape == (2, 4)
    assert output.retrieval_logits.shape == (2, 64)
    assert output.retrieved_intervals.shape == (2, 4, 2)
    assert output.boundary_probabilities.shape == (2, 40)
    assert torch.isfinite(loss.total)
    if condition is MemoryCondition.CONTEXT_ONLY:
        assert not output.retrieval_mask.any()
        assert torch.all(output.retrieval_targets == -100)


def test_local_temporal_context_cannot_carry_earlier_block_state() -> None:
    batch, vocabulary = tensorize_examples(_episode_query_pairs(), max_entities=8)
    model = _model(vocabulary).eval()
    original = model(batch, condition=MemoryCondition.CONTEXT_ONLY)

    changed_positions = batch.sequence.agent_position.clone()
    changed_positions[:, 0] = torch.tensor([5.0, 3.0])
    changed_sequence = replace(batch.sequence, agent_position=changed_positions)
    changed_batch = replace(batch, sequence=changed_sequence)
    changed = model(changed_batch, condition=MemoryCondition.CONTEXT_ONLY)

    assert not torch.equal(
        original.temporal_states[:, 0], changed.temporal_states[:, 0]
    )
    assert torch.equal(
        original.temporal_states[:, -8:], changed.temporal_states[:, -8:]
    )


def test_one_parameterization_is_reused_across_conditions() -> None:
    batch, vocabulary = tensorize_examples(_episode_query_pairs(), max_entities=8)
    model = _model(vocabulary)
    parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}

    for condition in MemoryCondition:
        model(batch, condition=condition, random_seed=5)
        assert parameter_ids == {
            name: id(parameter) for name, parameter in model.named_parameters()
        }
