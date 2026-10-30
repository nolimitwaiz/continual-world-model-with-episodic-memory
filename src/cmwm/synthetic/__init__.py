"""Deterministic symbolic-world generation for Prototype 1A."""

from cmwm.synthetic.audit import (
    CausalCandidateControl,
    G0AuditPolicy,
    G0AuditReport,
    audit_g0,
    summarize_query_balance,
    summarize_segment_length_match,
    summarize_storage_pressure,
)
from cmwm.synthetic.generator import SyntheticEpisodeGenerator
from cmwm.synthetic.causal_controls import (
    causal_candidate_controls,
    causal_control_is_matched,
    causal_controls_for_dataset,
)
from cmwm.synthetic.manifest import (
    DatasetManifest,
    ManifestSeal,
    SidecarManifestEntry,
    assert_manifest_sealed,
    build_dataset_manifest,
    seal_manifest,
    verify_manifest_seal,
)
from cmwm.synthetic.queries import answer_query, generate_queries
from cmwm.synthetic.validation import ValidationError, episode_checksum, validate_episode
from cmwm.synthetic.world import InvalidActionError, SymbolicWorld, observe, simulate_step

__all__ = [
    "CausalCandidateControl",
    "DatasetManifest",
    "G0AuditPolicy",
    "G0AuditReport",
    "InvalidActionError",
    "ManifestSeal",
    "SidecarManifestEntry",
    "SymbolicWorld",
    "SyntheticEpisodeGenerator",
    "ValidationError",
    "assert_manifest_sealed",
    "audit_g0",
    "answer_query",
    "build_dataset_manifest",
    "causal_candidate_controls",
    "causal_control_is_matched",
    "causal_controls_for_dataset",
    "episode_checksum",
    "generate_queries",
    "observe",
    "seal_manifest",
    "simulate_step",
    "summarize_query_balance",
    "summarize_segment_length_match",
    "summarize_storage_pressure",
    "validate_episode",
    "verify_manifest_seal",
]
