"""Deterministic seed controls shared by future trainer implementations."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - required in the cluster environment
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - required in the cluster environment
        pass
