"""Fixed temporal baseline for every CMWM-E1 comparison condition."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from cmwm.models.contracts import ParameterAccountable
from cmwm.models.symbolic import SYMBOLIC_STATE_DIM


TEMPORAL_HIDDEN_DIM = 128


@dataclass(frozen=True)
class TemporalCoreOutput:
    sequence: Tensor
    final_state: Tensor


class FixedGRUTemporalCore(ParameterAccountable, nn.Module):
    """One-layer, 128-unit GRU held identical across C0--C5."""

    def __init__(
        self,
        *,
        input_dim: int = SYMBOLIC_STATE_DIM,
        hidden_dim: int = TEMPORAL_HIDDEN_DIM,
        layers: int = 1,
    ) -> None:
        super().__init__()
        if input_dim != SYMBOLIC_STATE_DIM:
            raise ValueError("CMWM-E1 fixes the temporal input dimension at 128")
        if hidden_dim != TEMPORAL_HIDDEN_DIM:
            raise ValueError("CMWM-E1 fixes the GRU hidden dimension at 128")
        if layers != 1:
            raise ValueError("CMWM-E1 fixes the primary GRU to one layer")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=0.0,
        )

    def forward(
        self, sequence: Tensor, initial_state: Tensor | None = None
    ) -> TemporalCoreOutput:
        if sequence.ndim != 3 or sequence.shape[-1] != self.input_dim:
            raise ValueError("sequence must have shape [batch, steps, 128]")
        if initial_state is not None:
            expected = (self.layers, sequence.shape[0], self.hidden_dim)
            if tuple(initial_state.shape) != expected:
                raise ValueError(f"initial_state must have shape {expected}")
        output, final_state = self.gru(sequence, initial_state)
        return TemporalCoreOutput(sequence=output, final_state=final_state)

