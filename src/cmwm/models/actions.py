"""Condition-blind action encoding for the symbolic CMWM-E1 baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from cmwm.models.contracts import ParameterAccountable


ACTION_KIND_COUNT = 9
ACTION_DIM = 128


@dataclass(frozen=True)
class ActionBatch:
    """Observed action fields only; no effect, outcome, or causal annotation."""

    kind: Tensor
    target_entity_slot: Tensor
    container_slot: Tensor
    destination: Tensor
    destination_mask: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.kind.shape[0])

    def validate(self, *, max_entities: int) -> None:
        expected = (self.batch_size,)
        for name, tensor in (
            ("kind", self.kind),
            ("target_entity_slot", self.target_entity_slot),
            ("container_slot", self.container_slot),
            ("destination_mask", self.destination_mask),
        ):
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must have shape [batch]")
        if tuple(self.destination.shape) != (self.batch_size, 2):
            raise ValueError("destination must have shape [batch, 2]")
        if self.destination_mask.dtype is not torch.bool:
            raise TypeError("destination_mask must be boolean")
        if torch.any(self.kind < 0) or torch.any(self.kind >= ACTION_KIND_COUNT):
            raise ValueError("action kind is outside the fixed action vocabulary")
        for name, slots in (
            ("target_entity_slot", self.target_entity_slot),
            ("container_slot", self.container_slot),
        ):
            if torch.any(slots < -1) or torch.any(slots >= max_entities):
                raise ValueError(f"{name} is outside the entity-slot budget")


class CanonicalActionEncoder(ParameterAccountable, nn.Module):
    """Encode an observed action without encoding its oracle effect."""

    def __init__(
        self,
        *,
        max_entities: int,
        grid_width: int,
        grid_height: int,
        output_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        if max_entities <= 0:
            raise ValueError("max_entities must be positive")
        if grid_width <= 1 or grid_height <= 1:
            raise ValueError("grid dimensions must exceed one")
        if output_dim != ACTION_DIM:
            raise ValueError("CMWM-E1 fixes the action representation at 128")
        self.max_entities = max_entities
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.kind_embedding = nn.Embedding(ACTION_KIND_COUNT, 8)
        self.target_embedding = nn.Embedding(max_entities + 1, 16)
        self.container_embedding = nn.Embedding(max_entities + 1, 16)
        self.projection = nn.Sequential(
            nn.Linear(43, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, action: ActionBatch) -> Tensor:
        action.validate(max_entities=self.max_entities)
        scale = action.destination.new_tensor(
            (self.grid_width - 1, self.grid_height - 1)
        )
        destination = action.destination.to(self.kind_embedding.weight.dtype) / scale
        destination = destination * action.destination_mask.unsqueeze(-1).to(
            destination.dtype
        )
        features = torch.cat(
            (
                self.kind_embedding(action.kind),
                self.target_embedding(action.target_entity_slot + 1),
                self.container_embedding(action.container_slot + 1),
                destination,
                action.destination_mask.to(destination.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        return self.projection(features)

