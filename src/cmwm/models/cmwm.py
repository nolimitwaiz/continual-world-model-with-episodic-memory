"""Integrated Continual Multimodal World Model, symbolic E1 path.

CMWM is one model.  Event-structured episodic retrieval is a native subsystem,
not a second product.  The subsystem remains externally selectable only so the
scientific protocol can compare it with context, frame, and non-semantic chunk
controls while every learned component stays identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cmwm.data import (
    CAUSAL_TARGET_KINDS,
    TARGET_FIELDS,
    AnswerVocabulary,
    CMWMBatch,
)
from cmwm.models.boundaries import (
    FixedWindowSegmenter,
    OracleSegmenter,
    RandomMatchedSegmenter,
    Segment,
    segments_from_boundaries,
)
from cmwm.models.contracts import finite_mask_value
from cmwm.models.memory_writer import (
    MAX_RETRIEVED_RECORDS,
    MEMORY_RECORD_CAPACITY,
    SharedKeyValueMemoryWriter,
)
from cmwm.models.queries import (
    QueryHeadOutput,
    RetrievedMemoryBatch,
    SharedQueryHead,
    StructuredQueryEncoder,
)
from cmwm.models.symbolic import CanonicalSymbolicStepAdapter, SYMBOLIC_STATE_DIM
from cmwm.models.temporal import FixedGRUTemporalCore


class MemoryCondition(str, Enum):
    CONTEXT_ONLY = "context_only"
    FRAME = "frame_memory"
    FIXED = "fixed_chunks"
    RANDOM_MATCHED = "random_matched"
    ORACLE_EVENTS = "oracle_events"
    LEARNED_EVENTS = "learned_events"
    PREDICTION_ERROR = "prediction_error"


@dataclass(frozen=True)
class CMWMOutput:
    answer_logits: Tensor
    candidate_logits: Tensor
    candidate_targets: Tensor
    retrieval_logits: Tensor
    retrieval_targets: Tensor
    retrieval_mask: Tensor
    retrieved_intervals: Tensor
    boundary_logits: Tensor
    boundary_probabilities: Tensor
    temporal_states: Tensor
    next_state_predictions: Tensor


@dataclass(frozen=True)
class CMWMLoss:
    total: Tensor
    answer: Tensor
    retrieval: Tensor
    causal_pointer: Tensor
    boundary: Tensor
    prediction: Tensor


class LearnedBoundaryDetector(nn.Module):
    """Online transition detector using only current and prior latent states."""

    def __init__(self, hidden_dim: int = SYMBOLIC_STATE_DIM) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, temporal_states: Tensor) -> Tensor:
        if temporal_states.ndim != 3 or temporal_states.shape[-1] != SYMBOLIC_STATE_DIM:
            raise ValueError("temporal_states must have shape [batch, steps, 128]")
        previous = torch.cat(
            (torch.zeros_like(temporal_states[:, :1]), temporal_states[:, :-1]), dim=1
        )
        features = torch.cat(
            (temporal_states, previous, (temporal_states - previous).abs()), dim=-1
        )
        logits = self.network(features).squeeze(-1)
        return logits


class CMWM(nn.Module):
    """The single integrated CMWM-E1 model with an ablatable memory policy.

    The GRU is reset at fixed local-context boundaries.  This prevents its
    hidden state from becoming an undeclared full-episode memory.  Episodic
    records are written from the resulting condition-independent latent stream.
    """

    def __init__(
        self,
        *,
        max_entities: int,
        grid_width: int,
        grid_height: int,
        context_steps: int,
        answer_vocabulary: AnswerVocabulary,
        fixed_chunk_steps: int = 8,
        boundary_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if context_steps <= 0:
            raise ValueError("context_steps must be positive")
        if fixed_chunk_steps <= 0:
            raise ValueError("fixed_chunk_steps must be positive")
        if not 0.0 <= boundary_threshold <= 1.0:
            raise ValueError("boundary_threshold must lie in [0, 1]")
        if (
            answer_vocabulary.grid_width != grid_width
            or answer_vocabulary.grid_height != grid_height
            or answer_vocabulary.max_entities != max_entities
        ):
            raise ValueError("answer vocabulary and model geometry disagree")

        self.max_entities = max_entities
        self.context_steps = context_steps
        self.fixed_chunk_steps = fixed_chunk_steps
        self.boundary_threshold = boundary_threshold
        self.answer_vocabulary = answer_vocabulary
        self.step_adapter = CanonicalSymbolicStepAdapter(
            max_entities=max_entities,
            grid_width=grid_width,
            grid_height=grid_height,
        )
        self.temporal_core = FixedGRUTemporalCore()
        self.memory_writer = SharedKeyValueMemoryWriter()
        self.query_encoder = StructuredQueryEncoder(
            max_entities=max_entities,
            field_vocab_size=len(TARGET_FIELDS),
            causal_target_vocab_size=len(CAUSAL_TARGET_KINDS),
            answer_vocab_size=answer_vocabulary.size,
        )
        self.query_head = SharedQueryHead(answer_vocab_size=answer_vocabulary.size)
        self.boundary_detector = LearnedBoundaryDetector()
        self.next_state_predictor = nn.Linear(SYMBOLIC_STATE_DIM, SYMBOLIC_STATE_DIM)

    def _local_temporal_states(self, batch: CMWMBatch) -> Tensor:
        sequence = batch.sequence
        encoded = self.step_adapter(
            sequence.flattened_state(), sequence.flattened_action()
        ).reshape(sequence.batch_size, sequence.step_count, SYMBOLIC_STATE_DIM)
        blocks: list[Tensor] = []
        for start in range(0, sequence.step_count, self.context_steps):
            end = min(start + self.context_steps, sequence.step_count)
            # Hidden state intentionally resets here.
            blocks.append(self.temporal_core(encoded[:, start:end]).sequence)
        temporal = torch.cat(blocks, dim=1)
        return temporal * sequence.step_mask.unsqueeze(-1).to(temporal.dtype)

    def _boundary_scores(self, temporal: Tensor, step_mask: Tensor) -> tuple[Tensor, Tensor]:
        raw_logits = self.boundary_detector(temporal)
        # Position ``s`` means "close a record after world transition s".
        # The decision therefore uses the latent for zero-based item ``s - 1``.
        learned_logits = torch.full_like(raw_logits, finite_mask_value(raw_logits))
        learned_logits[:, 1:] = raw_logits[:, :-1]
        learned_logits = learned_logits.masked_fill(
            ~step_mask, finite_mask_value(learned_logits)
        )
        learned_probabilities = torch.sigmoid(learned_logits)
        return learned_logits, learned_probabilities

    def _prediction_error_probabilities(
        self, temporal: Tensor, step_mask: Tensor
    ) -> Tensor:
        predicted = self.next_state_predictor(temporal[:, :-1])
        squared_error = (predicted - temporal[:, 1:]).square().mean(dim=-1)
        valid = step_mask[:, 1:]
        probabilities = torch.zeros_like(temporal[..., 0])
        for index in range(temporal.shape[0]):
            values = squared_error[index, valid[index]]
            if values.numel() == 0:
                continue
            normalized = (squared_error[index] - values.mean()) / values.std(
                unbiased=False
            ).clamp_min(1.0e-6)
            # Error j predicts item j + 1 (world step j + 2), so it controls
            # the split position j + 2.  The first transition has no prior
            # latent from which to form a prediction.
            probabilities[index, 2:] = torch.sigmoid(normalized[:-1])
        return probabilities * step_mask.to(probabilities.dtype)

    @staticmethod
    def _oracle_boundaries(batch: CMWMBatch, index: int, length: int) -> tuple[int, ...]:
        return tuple(
            int(step)
            for step in torch.nonzero(batch.oracle_boundaries[index], as_tuple=False)
            .flatten()
            .tolist()
            if 0 < int(step) < length
        )

    def _segments(
        self,
        *,
        condition: MemoryCondition,
        batch: CMWMBatch,
        batch_index: int,
        length: int,
        learned_probabilities: Tensor,
        prediction_error_probabilities: Tensor,
        random_seed: int,
    ) -> tuple[Segment, ...]:
        if condition is MemoryCondition.CONTEXT_ONLY:
            return ()
        if condition is MemoryCondition.FRAME:
            boundaries = tuple(range(1, length))
        elif condition is MemoryCondition.FIXED:
            boundaries = FixedWindowSegmenter(self.fixed_chunk_steps).boundaries(
                length=length
            )
        elif condition is MemoryCondition.RANDOM_MATCHED:
            oracle = self._oracle_boundaries(batch, batch_index, length)
            oracle_segments = segments_from_boundaries(length, oracle)
            boundaries = RandomMatchedSegmenter(
                tuple(segment.length for segment in oracle_segments),
                seed=random_seed + batch_index,
            ).boundaries(length=length)
        elif condition is MemoryCondition.ORACLE_EVENTS:
            boundaries = OracleSegmenter(
                self._oracle_boundaries(batch, batch_index, length)
            ).boundaries(length=length)
        elif condition is MemoryCondition.LEARNED_EVENTS:
            boundaries = tuple(
                step
                for step in range(1, length)
                if float(learned_probabilities[batch_index, step].detach())
                >= self.boundary_threshold
            )
        elif condition is MemoryCondition.PREDICTION_ERROR:
            boundaries = tuple(
                step
                for step in range(1, length)
                if float(prediction_error_probabilities[batch_index, step].detach())
                >= self.boundary_threshold
            )
        else:  # pragma: no cover - exhaustive Enum guard
            raise ValueError(f"unsupported memory condition: {condition}")
        return segments_from_boundaries(length, boundaries)

    def forward(
        self,
        batch: CMWMBatch,
        *,
        condition: MemoryCondition | str,
        random_seed: int = 0,
    ) -> CMWMOutput:
        batch.validate()
        condition = MemoryCondition(condition)
        temporal = self._local_temporal_states(batch)
        boundary_logits, boundary_probabilities = self._boundary_scores(
            temporal, batch.sequence.step_mask
        )
        prediction_error_probabilities = self._prediction_error_probabilities(
            temporal, batch.sequence.step_mask
        )
        query_encoding = self.query_encoder(batch.query)

        current_states: list[Tensor] = []
        retrieved_keys: list[Tensor] = []
        retrieved_values: list[Tensor] = []
        retrieved_masks: list[Tensor] = []
        retrieved_intervals: list[Tensor] = []
        retrieval_logits: list[Tensor] = []
        retrieval_masks: list[Tensor] = []
        retrieval_targets: list[int] = []
        candidate_targets: list[int] = []

        for index in range(batch.batch_size):
            length = int(batch.ask_steps[index])
            current_states.append(temporal[index, length - 1])
            segments = self._segments(
                condition=condition,
                batch=batch,
                batch_index=index,
                length=length,
                learned_probabilities=boundary_probabilities,
                prediction_error_probabilities=prediction_error_probabilities,
                random_seed=random_seed,
            )
            # FIFO byte capacity is equivalent to 64 records under the fixed layout.
            segments = segments[-MEMORY_RECORD_CAPACITY:]
            if segments:
                keys_for_episode: list[Tensor] = []
                values_for_episode: list[Tensor] = []
                for segment in segments:
                    states = temporal[index : index + 1, segment.start : segment.end]
                    mask = torch.ones(
                        states.shape[:2], device=states.device, dtype=torch.bool
                    )
                    written = self.memory_writer(states, mask)
                    keys_for_episode.append(written.keys[0])
                    values_for_episode.append(written.values[0])
                all_keys = torch.stack(keys_for_episode)
                all_values = torch.stack(values_for_episode)
                all_scores = torch.mv(all_keys, query_encoding[index]) / sqrt(
                    SYMBOLIC_STATE_DIM
                )
                requested = min(MAX_RETRIEVED_RECORDS, len(segments))
                selected = torch.topk(all_scores, k=requested).indices
                selected_keys = all_keys[selected]
                selected_values = all_values[selected]
                selected_segments = [segments[int(item)] for item in selected]
                reference = (
                    int(batch.causal_source_steps[index])
                    if int(batch.causal_source_steps[index]) >= 0
                    else int(batch.reference_steps[index])
                )
                target_record = next(
                    (
                        record_index
                        for record_index, segment in enumerate(segments)
                        if segment.start < reference <= segment.end
                    ),
                    -100,
                )
                selected_target = next(
                    (
                        selected_index
                        for selected_index, segment in enumerate(selected_segments)
                        if segment.start < reference <= segment.end
                    ),
                    -100,
                )
            else:
                all_scores = temporal.new_empty((0,))
                selected_keys = temporal.new_empty((0, SYMBOLIC_STATE_DIM))
                selected_values = temporal.new_empty((0, SYMBOLIC_STATE_DIM))
                selected_segments = []
                target_record = -100
                selected_target = -100

            retrieved_count = int(selected_keys.shape[0])
            pad_count = MAX_RETRIEVED_RECORDS - retrieved_count
            retrieved_keys.append(
                F.pad(selected_keys, (0, 0, 0, pad_count), value=0.0)
            )
            retrieved_values.append(
                F.pad(selected_values, (0, 0, 0, pad_count), value=0.0)
            )
            retrieved_masks.append(
                torch.arange(MAX_RETRIEVED_RECORDS, device=temporal.device)
                < retrieved_count
            )
            intervals = torch.full(
                (MAX_RETRIEVED_RECORDS, 2),
                -1,
                device=temporal.device,
                dtype=torch.long,
            )
            for selected_index, segment in enumerate(selected_segments):
                intervals[selected_index] = torch.tensor(
                    (segment.start, segment.end), device=temporal.device
                )
            retrieved_intervals.append(intervals)
            score_pad = MEMORY_RECORD_CAPACITY - int(all_scores.shape[0])
            retrieval_logits.append(
                F.pad(
                    all_scores,
                    (0, score_pad),
                    value=finite_mask_value(all_scores),
                )
            )
            retrieval_masks.append(
                torch.arange(MEMORY_RECORD_CAPACITY, device=temporal.device)
                < int(all_scores.shape[0])
            )
            retrieval_targets.append(target_record)
            candidate_targets.append(selected_target)

        current = torch.stack(current_states)
        retrieved = RetrievedMemoryBatch(
            keys=torch.stack(retrieved_keys),
            values=torch.stack(retrieved_values),
            mask=torch.stack(retrieved_masks),
        )
        head: QueryHeadOutput = self.query_head(current, query_encoding, retrieved)
        return CMWMOutput(
            answer_logits=head.answer_logits,
            candidate_logits=head.candidate_logits,
            candidate_targets=torch.tensor(
                candidate_targets, device=temporal.device, dtype=torch.long
            ),
            retrieval_logits=torch.stack(retrieval_logits),
            retrieval_targets=torch.tensor(
                retrieval_targets, device=temporal.device, dtype=torch.long
            ),
            retrieval_mask=torch.stack(retrieval_masks),
            retrieved_intervals=torch.stack(retrieved_intervals),
            boundary_logits=boundary_logits,
            boundary_probabilities=(
                prediction_error_probabilities
                if condition is MemoryCondition.PREDICTION_ERROR
                else boundary_probabilities
            ),
            temporal_states=temporal,
            next_state_predictions=self.next_state_predictor(temporal[:, :-1]),
        )

    @staticmethod
    def loss(
        output: CMWMOutput,
        batch: CMWMBatch,
        *,
        condition: MemoryCondition | str,
        retrieval_weight: float = 1.0,
        causal_pointer_weight: float = 1.0,
        boundary_weight: float = 1.0,
        prediction_weight: float = 0.1,
    ) -> CMWMLoss:
        condition = MemoryCondition(condition)
        zero = output.answer_logits.sum() * 0.0

        answer_mask = batch.answer_targets >= 0
        answer = (
            F.cross_entropy(
                output.answer_logits[answer_mask], batch.answer_targets[answer_mask]
            )
            if torch.any(answer_mask)
            else zero
        )
        retrieval_mask = output.retrieval_targets >= 0
        retrieval = (
            F.cross_entropy(
                output.retrieval_logits[retrieval_mask],
                output.retrieval_targets[retrieval_mask],
            )
            if torch.any(retrieval_mask)
            else zero
        )
        causal_mask = (batch.causal_source_steps >= 0) & (
            output.candidate_targets >= 0
        )
        causal_pointer = (
            F.cross_entropy(
                output.candidate_logits[causal_mask],
                output.candidate_targets[causal_mask],
            )
            if torch.any(causal_mask)
            else zero
        )
        valid_boundary_positions = batch.sequence.step_mask.clone()
        valid_boundary_positions[:, 0] = False
        if condition is MemoryCondition.LEARNED_EVENTS and torch.any(
            valid_boundary_positions
        ):
            targets = batch.oracle_boundaries.to(output.boundary_logits.dtype)
            positives = targets[valid_boundary_positions].sum()
            negatives = valid_boundary_positions.sum() - positives
            positive_weight = (negatives / positives.clamp_min(1.0)).detach()
            boundary = F.binary_cross_entropy_with_logits(
                output.boundary_logits[valid_boundary_positions],
                targets[valid_boundary_positions],
                pos_weight=positive_weight,
            )
        else:
            boundary = zero
        prediction_positions = (
            batch.sequence.step_mask[:, :-1] & batch.sequence.step_mask[:, 1:]
        )
        prediction = (
            F.mse_loss(
                output.next_state_predictions[prediction_positions],
                output.temporal_states[:, 1:][prediction_positions].detach(),
            )
            if torch.any(prediction_positions)
            else zero
        )
        total = (
            answer
            + retrieval_weight * retrieval
            + causal_pointer_weight * causal_pointer
            + boundary_weight * boundary
            + prediction_weight * prediction
        )
        return CMWMLoss(
            total=total,
            answer=answer,
            retrieval=retrieval,
            causal_pointer=causal_pointer,
            boundary=boundary,
            prediction=prediction,
        )
