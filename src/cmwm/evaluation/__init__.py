"""Evaluation utilities and pre-registered decision gates for Prototype 1."""

from cmwm.evaluation.gates import (
    Budget,
    GateResult,
    GateThresholds,
    PairedEffect,
    evaluate_detector_gate,
    evaluate_oracle_gate,
    verify_matched_budgets,
)
from cmwm.evaluation.metrics import BoundaryMetrics, boundary_metrics
from cmwm.evaluation.conformance import (
    EndpointRow,
    effective_query_lag,
    endpoint_rows,
    summarize_endpoint_rows,
)
from cmwm.evaluation.replication import (
    EXPECTED_SEEDS,
    aggregate_development_replication,
    paired_development_effect,
)

__all__ = [
    "BoundaryMetrics",
    "Budget",
    "EndpointRow",
    "EXPECTED_SEEDS",
    "GateResult",
    "GateThresholds",
    "PairedEffect",
    "boundary_metrics",
    "effective_query_lag",
    "endpoint_rows",
    "aggregate_development_replication",
    "evaluate_detector_gate",
    "evaluate_oracle_gate",
    "verify_matched_budgets",
    "paired_development_effect",
    "summarize_endpoint_rows",
]
