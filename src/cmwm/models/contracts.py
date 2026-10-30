"""Shared model-accounting contracts for the fixed CMWM-E1 components."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ParameterAccounting:
    """Auditable parameter totals used by the equal-budget checks."""

    total: int
    trainable: int

    @property
    def frozen(self) -> int:
        return self.total - self.trainable

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "trainable": self.trainable,
            "frozen": self.frozen,
        }


class ParameterAccountable:
    """Mixin giving every shared component one parameter-counting rule."""

    def parameter_accounting(self) -> ParameterAccounting:
        if not isinstance(self, nn.Module):  # pragma: no cover - defensive misuse guard
            raise TypeError("ParameterAccountable must be mixed into torch.nn.Module")
        parameters = tuple(self.parameters())
        return ParameterAccounting(
            total=sum(parameter.numel() for parameter in parameters),
            trainable=sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            ),
        )


def finite_mask_value(tensor: Tensor) -> float:
    """Return a finite masking sentinel representable by the tensor dtype."""

    if not tensor.dtype.is_floating_point:
        raise TypeError("masking sentinel requires a floating-point tensor")
    return float(torch.finfo(tensor.dtype).min)
