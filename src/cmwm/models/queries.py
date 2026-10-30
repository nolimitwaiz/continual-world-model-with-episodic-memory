"""Condition-blind structured-query encoder and shared answer head."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn

from cmwm.models.contracts import ParameterAccountable, finite_mask_value
from cmwm.models.symbolic import SYMBOLIC_STATE_DIM


QUERY_KIND_COUNT = 4
QUERY_DIM = 128
MAX_RETRIEVED_RECORDS = 4


@dataclass(frozen=True)
class QueryBatch:
    """Safe tensor query excluding its oracle answer and candidate identity.

    For causal-predecessor queries, ``causal_target_kind`` and
    ``causal_target_entity_slot`` describe the outcome named by the query. They
    are not causal edges. ``-1`` is used for non-causal query types.
    """

    kind: Tensor
    target_entity_slot: Tensor
    target_field: Tensor
    lag: Tensor
    causal_target_kind: Tensor
    causal_target_entity_slot: Tensor
    causal_value_token: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.kind.shape[0])

    def validate(
        self,
        *,
        max_entities: int,
        field_vocab_size: int,
        causal_target_vocab_size: int,
        answer_vocab_size: int,
    ) -> None:
        expected = (self.batch_size,)
        for name, tensor in (
            ("kind", self.kind),
            ("target_entity_slot", self.target_entity_slot),
            ("target_field", self.target_field),
            ("lag", self.lag),
            ("causal_target_kind", self.causal_target_kind),
            ("causal_target_entity_slot", self.causal_target_entity_slot),
            ("causal_value_token", self.causal_value_token),
        ):
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must have shape [batch]")
        if torch.any(self.kind < 0) or torch.any(self.kind >= QUERY_KIND_COUNT):
            raise ValueError("query kind is outside the fixed task vocabulary")
        if torch.any(self.target_entity_slot < -1) or torch.any(
            self.target_entity_slot >= max_entities
        ):
            raise ValueError("query target entity is outside the slot budget")
        if torch.any(self.target_field < -1) or torch.any(
            self.target_field >= field_vocab_size
        ):
            raise ValueError("query target field is outside the field vocabulary")
        if torch.any(self.lag < 0):
            raise ValueError("query lag cannot be negative")
        if torch.any(self.causal_target_kind < -1) or torch.any(
            self.causal_target_kind >= causal_target_vocab_size
        ):
            raise ValueError("causal target kind is outside the declared vocabulary")
        if torch.any(self.causal_target_entity_slot < -1) or torch.any(
            self.causal_target_entity_slot >= max_entities
        ):
            raise ValueError("causal target entity is outside the entity-slot budget")
        if torch.any(self.causal_value_token < -1) or torch.any(
            self.causal_value_token >= answer_vocab_size
        ):
            raise ValueError("causal value is outside the fixed answer vocabulary")


@dataclass(frozen=True)
class RetrievedMemoryBatch:
    keys: Tensor
    values: Tensor
    mask: Tensor

    def validate(self, *, batch_size: int) -> None:
        if self.keys.ndim != 3 or self.values.ndim != 3:
            raise ValueError("retrieved keys and values must be rank-three tensors")
        if self.keys.shape != self.values.shape:
            raise ValueError("retrieved keys and values must have equal shapes")
        if self.keys.shape[0] != batch_size or self.keys.shape[2] != QUERY_DIM:
            raise ValueError("retrieved records must have shape [batch, records, 128]")
        if self.keys.shape[1] > MAX_RETRIEVED_RECORDS:
            raise ValueError("retrieval exceeds the preregistered top-four budget")
        if tuple(self.mask.shape) != tuple(self.keys.shape[:2]):
            raise ValueError("retrieval mask must have shape [batch, records]")
        if self.mask.dtype is not torch.bool:
            raise TypeError("retrieval mask must be boolean")

    @classmethod
    def empty(
        cls,
        *,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
        slots: int = MAX_RETRIEVED_RECORDS,
    ) -> "RetrievedMemoryBatch":
        if slots < 0 or slots > MAX_RETRIEVED_RECORDS:
            raise ValueError("slots must lie between zero and four")
        shape = (batch_size, slots, QUERY_DIM)
        return cls(
            keys=torch.zeros(shape, device=device, dtype=dtype),
            values=torch.zeros(shape, device=device, dtype=dtype),
            mask=torch.zeros((batch_size, slots), device=device, dtype=torch.bool),
        )


class StructuredQueryEncoder(ParameterAccountable, nn.Module):
    """Encode the same non-linguistic query schema for every condition."""

    def __init__(
        self,
        *,
        max_entities: int,
        field_vocab_size: int,
        causal_target_vocab_size: int,
        answer_vocab_size: int = 64,
        output_dim: int = QUERY_DIM,
    ) -> None:
        super().__init__()
        if (
            max_entities <= 0
            or field_vocab_size <= 0
            or causal_target_vocab_size <= 0
            or answer_vocab_size <= 1
        ):
            raise ValueError("query vocabularies must be positive")
        if output_dim != QUERY_DIM:
            raise ValueError("CMWM-E1 fixes the query representation at 128")
        self.max_entities = max_entities
        self.field_vocab_size = field_vocab_size
        self.causal_target_vocab_size = causal_target_vocab_size
        self.answer_vocab_size = answer_vocab_size
        self.kind_embedding = nn.Embedding(QUERY_KIND_COUNT, 8)
        self.entity_embedding = nn.Embedding(max_entities + 1, 16)
        self.field_embedding = nn.Embedding(field_vocab_size + 1, 8)
        self.causal_target_embedding = nn.Embedding(causal_target_vocab_size + 1, 8)
        self.causal_entity_embedding = nn.Embedding(max_entities + 1, 16)
        self.causal_value_embedding = nn.Embedding(answer_vocab_size + 1, 16)
        self.projection = nn.Sequential(
            nn.Linear(74, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, query: QueryBatch) -> Tensor:
        query.validate(
            max_entities=self.max_entities,
            field_vocab_size=self.field_vocab_size,
            causal_target_vocab_size=self.causal_target_vocab_size,
            answer_vocab_size=self.answer_vocab_size,
        )
        lag = query.lag.to(self.kind_embedding.weight.dtype)
        lag_features = torch.stack((torch.log1p(lag), 1.0 / (lag + 1.0)), dim=-1)
        features = torch.cat(
            (
                self.kind_embedding(query.kind),
                self.entity_embedding(query.target_entity_slot + 1),
                self.field_embedding(query.target_field + 1),
                self.causal_target_embedding(query.causal_target_kind + 1),
                self.causal_entity_embedding(query.causal_target_entity_slot + 1),
                self.causal_value_embedding(query.causal_value_token + 1),
                lag_features,
            ),
            dim=-1,
        )
        return self.projection(features)


@dataclass(frozen=True)
class QueryHeadOutput:
    """Fixed-vocabulary answers plus opaque pointer scores over retrieved records."""

    answer_logits: Tensor
    candidate_logits: Tensor


class SharedQueryHead(ParameterAccountable, nn.Module):
    """One answer head for C0--C5 with a top-four memory interface.

    ``candidate_logits`` are pointer scores over the retrieved-record axis.
    Opaque record identifiers remain outside the model and are used only to map
    the selected index to the evaluator's candidate identifier.
    """

    def __init__(self, *, answer_vocab_size: int) -> None:
        super().__init__()
        if answer_vocab_size <= 1:
            raise ValueError("answer_vocab_size must exceed one")
        self.answer_vocab_size = answer_vocab_size
        self.output = nn.Sequential(
            nn.Linear(SYMBOLIC_STATE_DIM + QUERY_DIM + QUERY_DIM, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, answer_vocab_size),
        )
        self.candidate_query = nn.Linear(SYMBOLIC_STATE_DIM + QUERY_DIM, QUERY_DIM)

    def forward(
        self,
        current_state: Tensor,
        query_encoding: Tensor,
        retrieved: RetrievedMemoryBatch,
    ) -> QueryHeadOutput:
        if current_state.ndim != 2 or current_state.shape[1] != SYMBOLIC_STATE_DIM:
            raise ValueError("current_state must have shape [batch, 128]")
        if tuple(query_encoding.shape) != tuple(current_state.shape):
            raise ValueError("query_encoding must have shape [batch, 128]")
        retrieved.validate(batch_size=int(current_state.shape[0]))

        scores = torch.einsum("bd,bkd->bk", query_encoding, retrieved.keys)
        scores = scores / sqrt(QUERY_DIM)
        masked_scores = scores.masked_fill(~retrieved.mask, finite_mask_value(scores))
        weights = torch.softmax(masked_scores, dim=-1) * retrieved.mask.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        memory_summary = torch.einsum("bk,bkd->bd", weights, retrieved.values)
        answer_logits = self.output(
            torch.cat((current_state, query_encoding, memory_summary), dim=-1)
        )
        pointer_query = self.candidate_query(
            torch.cat((current_state, query_encoding), dim=-1)
        )
        candidate_logits = torch.einsum("bd,bkd->bk", pointer_query, retrieved.values)
        candidate_logits = candidate_logits / sqrt(QUERY_DIM)
        candidate_logits = candidate_logits.masked_fill(
            ~retrieved.mask, finite_mask_value(candidate_logits)
        )
        return QueryHeadOutput(
            answer_logits=answer_logits,
            candidate_logits=candidate_logits,
        )
