"""Fixed CMWM-E1 components and model-independent baselines.

Tensor modules are imported lazily so local-safe boundary and byte-accounting
checks do not require PyTorch. Executable tensor checks remain cluster-only.
"""

from importlib import import_module
from typing import Any

from cmwm.models.boundaries import (
    BoundaryFeatures,
    FixedWindowSegmenter,
    OracleSegmenter,
    RandomMatchedSegmenter,
    Segment,
    SurpriseThresholdSegmenter,
    heuristic_boundary_probability,
    segments_from_boundaries,
)
from cmwm.models.memory import FifoVectorMemory, MemoryRecord


_LAZY_IMPORTS = {
    "CMWM": "cmwm.models.cmwm",
    "CMWMLoss": "cmwm.models.cmwm",
    "CMWMOutput": "cmwm.models.cmwm",
    "LearnedBoundaryDetector": "cmwm.models.cmwm",
    "MemoryCondition": "cmwm.models.cmwm",
    "ACTION_DIM": "cmwm.models.actions",
    "ACTION_KIND_COUNT": "cmwm.models.actions",
    "ActionBatch": "cmwm.models.actions",
    "CanonicalActionEncoder": "cmwm.models.actions",
    "MEMORY_RECORD_CAPACITY": "cmwm.models.memory_writer",
    "MEMORY_VECTOR_DIM": "cmwm.models.memory_writer",
    "MemoryRecordLayout": "cmwm.models.memory_writer",
    "MemoryWriteBatch": "cmwm.models.memory_writer",
    "SharedKeyValueMemoryWriter": "cmwm.models.memory_writer",
    "MAX_RETRIEVED_RECORDS": "cmwm.models.queries",
    "QUERY_DIM": "cmwm.models.queries",
    "QueryBatch": "cmwm.models.queries",
    "QueryHeadOutput": "cmwm.models.queries",
    "RetrievedMemoryBatch": "cmwm.models.queries",
    "SharedQueryHead": "cmwm.models.queries",
    "StructuredQueryEncoder": "cmwm.models.queries",
    "SYMBOLIC_STATE_DIM": "cmwm.models.symbolic",
    "CanonicalSymbolicStepAdapter": "cmwm.models.symbolic",
    "CanonicalSymbolicStateAdapter": "cmwm.models.symbolic",
    "SymbolicStateBatch": "cmwm.models.symbolic",
    "TEMPORAL_HIDDEN_DIM": "cmwm.models.temporal",
    "FixedGRUTemporalCore": "cmwm.models.temporal",
    "TemporalCoreOutput": "cmwm.models.temporal",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value

__all__ = [
    "ACTION_DIM",
    "ACTION_KIND_COUNT",
    "ActionBatch",
    "BoundaryFeatures",
    "FixedWindowSegmenter",
    "FixedGRUTemporalCore",
    "FifoVectorMemory",
    "CanonicalActionEncoder",
    "CMWM",
    "CMWMLoss",
    "CMWMOutput",
    "CanonicalSymbolicStateAdapter",
    "CanonicalSymbolicStepAdapter",
    "MAX_RETRIEVED_RECORDS",
    "MEMORY_RECORD_CAPACITY",
    "MEMORY_VECTOR_DIM",
    "MemoryRecord",
    "MemoryRecordLayout",
    "MemoryCondition",
    "MemoryWriteBatch",
    "OracleSegmenter",
    "LearnedBoundaryDetector",
    "QUERY_DIM",
    "QueryBatch",
    "QueryHeadOutput",
    "RandomMatchedSegmenter",
    "RetrievedMemoryBatch",
    "SYMBOLIC_STATE_DIM",
    "Segment",
    "SharedKeyValueMemoryWriter",
    "SharedQueryHead",
    "StructuredQueryEncoder",
    "SurpriseThresholdSegmenter",
    "SymbolicStateBatch",
    "TEMPORAL_HIDDEN_DIM",
    "TemporalCoreOutput",
    "heuristic_boundary_probability",
    "segments_from_boundaries",
]
