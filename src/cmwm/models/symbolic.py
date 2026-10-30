"""Canonical, condition-blind symbolic-state interface for CMWM-E1 P1A."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from cmwm.models.actions import ActionBatch, CanonicalActionEncoder
from cmwm.models.contracts import ParameterAccountable


SYMBOLIC_STATE_DIM = 128
ENTITY_KIND_COUNT = 2


@dataclass(frozen=True)
class SymbolicStateBatch:
    """The complete and exclusive tensor input to the P1A state adapter.

    Entity slots are assigned independently of outcome, event, and comparison
    condition. ``-1`` is the sentinel for no held entity or no container.
    Event identifiers, boundary labels, causal edges, oracle answers, and
    condition identifiers intentionally have no fields in this contract.
    """

    agent_position: Tensor
    held_entity_slot: Tensor
    entity_positions: Tensor
    entity_kinds: Tensor
    contained_in_slot: Tensor
    entity_is_open: Tensor
    entity_mask: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.agent_position.shape[0])

    @property
    def entity_count(self) -> int:
        return int(self.entity_positions.shape[1])

    def validate(self, *, max_entities: int) -> None:
        if self.agent_position.ndim != 2 or self.agent_position.shape[1] != 2:
            raise ValueError("agent_position must have shape [batch, 2]")
        batch = self.batch_size
        if self.entity_positions.ndim != 3 or self.entity_positions.shape[2] != 2:
            raise ValueError("entity_positions must have shape [batch, entities, 2]")
        if self.entity_positions.shape[0] != batch:
            raise ValueError("symbolic tensors must have the same batch dimension")
        entities = self.entity_count
        if entities <= 0 or entities > max_entities:
            raise ValueError("entity dimension is outside the declared slot budget")
        expected = (batch, entities)
        for name, tensor in (
            ("entity_kinds", self.entity_kinds),
            ("contained_in_slot", self.contained_in_slot),
            ("entity_is_open", self.entity_is_open),
            ("entity_mask", self.entity_mask),
        ):
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must have shape [batch, entities]")
        if tuple(self.held_entity_slot.shape) != (batch,):
            raise ValueError("held_entity_slot must have shape [batch]")
        if self.entity_mask.dtype is not torch.bool:
            raise TypeError("entity_mask must be boolean")
        if torch.any(self.held_entity_slot < -1) or torch.any(
            self.held_entity_slot >= max_entities
        ):
            raise ValueError("held_entity_slot contains an invalid slot")
        if torch.any(self.contained_in_slot < -1) or torch.any(
            self.contained_in_slot >= max_entities
        ):
            raise ValueError("contained_in_slot contains an invalid slot")
        valid_kinds = self.entity_kinds[self.entity_mask]
        if torch.any(valid_kinds < 0) or torch.any(valid_kinds >= ENTITY_KIND_COUNT):
            raise ValueError("entity_kinds contains an unknown kind")


class CanonicalSymbolicStateAdapter(ParameterAccountable, nn.Module):
    """Map the safe symbolic schema to the fixed 128-dimensional interface."""

    def __init__(
        self,
        *,
        max_entities: int,
        grid_width: int,
        grid_height: int,
        output_dim: int = SYMBOLIC_STATE_DIM,
    ) -> None:
        super().__init__()
        if max_entities <= 0:
            raise ValueError("max_entities must be positive")
        if grid_width <= 1 or grid_height <= 1:
            raise ValueError("grid dimensions must exceed one")
        if output_dim != SYMBOLIC_STATE_DIM:
            raise ValueError("CMWM-E1 fixes the symbolic state dimension at 128")
        self.max_entities = max_entities
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.output_dim = output_dim

        self.kind_embedding = nn.Embedding(ENTITY_KIND_COUNT, 8)
        self.slot_embedding = nn.Embedding(max_entities, 16)
        self.container_embedding = nn.Embedding(max_entities + 1, 8)
        self.held_embedding = nn.Embedding(max_entities + 1, 16)
        self.entity_projection = nn.Sequential(
            nn.Linear(35, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )
        self.agent_projection = nn.Sequential(
            nn.Linear(18, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def _normalize_positions(self, positions: Tensor) -> Tensor:
        scale = positions.new_tensor((self.grid_width - 1, self.grid_height - 1))
        return positions.to(dtype=self.slot_embedding.weight.dtype) / scale

    def forward(self, batch: SymbolicStateBatch) -> Tensor:
        batch.validate(max_entities=self.max_entities)
        entity_count = batch.entity_count
        slots = torch.arange(entity_count, device=batch.entity_positions.device)
        slots = slots.unsqueeze(0).expand(batch.batch_size, -1)

        entity_features = torch.cat(
            (
                self._normalize_positions(batch.entity_positions),
                self.kind_embedding(batch.entity_kinds),
                self.container_embedding(batch.contained_in_slot + 1),
                batch.entity_is_open.to(self.slot_embedding.weight.dtype).unsqueeze(-1),
                self.slot_embedding(slots),
            ),
            dim=-1,
        )
        entity_states = self.entity_projection(entity_features)
        mask = batch.entity_mask.unsqueeze(-1).to(entity_states.dtype)
        entity_summary = (entity_states * mask).sum(dim=1)
        entity_summary = entity_summary / mask.sum(dim=1).clamp_min(1.0)

        agent_features = torch.cat(
            (
                self._normalize_positions(batch.agent_position),
                self.held_embedding(batch.held_entity_slot + 1),
            ),
            dim=-1,
        )
        agent_summary = self.agent_projection(agent_features)
        return self.output_projection(torch.cat((agent_summary, entity_summary), dim=-1))


class CanonicalSymbolicStepAdapter(ParameterAccountable, nn.Module):
    """Fuse current oracle state and the observed action into one safe step vector.

    The temporal core and memory writer must consume this representation rather
    than the state-only representation for action-effect and causal-predecessor
    tasks. Integration pairs ``StepRecord.action`` with ``StepRecord.after`` for
    the same completed transition; it must never pair an action with a later
    state. The fusion receives neither counterfactual outcomes nor causal edges.
    """

    def __init__(
        self,
        *,
        max_entities: int,
        grid_width: int,
        grid_height: int,
        output_dim: int = SYMBOLIC_STATE_DIM,
    ) -> None:
        super().__init__()
        if output_dim != SYMBOLIC_STATE_DIM:
            raise ValueError("CMWM-E1 fixes the symbolic step dimension at 128")
        self.state_adapter = CanonicalSymbolicStateAdapter(
            max_entities=max_entities,
            grid_width=grid_width,
            grid_height=grid_height,
            output_dim=output_dim,
        )
        self.action_encoder = CanonicalActionEncoder(
            max_entities=max_entities,
            grid_width=grid_width,
            grid_height=grid_height,
            output_dim=output_dim,
        )
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, state: SymbolicStateBatch, action: ActionBatch) -> Tensor:
        if state.batch_size != action.batch_size:
            raise ValueError("state and action must have equal batch sizes")
        return self.fusion(
            torch.cat((self.state_adapter(state), self.action_encoder(action)), dim=-1)
        )
