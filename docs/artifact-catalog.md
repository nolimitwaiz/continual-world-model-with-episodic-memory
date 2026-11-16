# Artifact catalog

This catalog indexes returned artifacts, not independent experiments. Many files are checkpoints of metadata, repeated summaries, or engineering audits.

## Current-project returned records

| Run or collection | JSON files | Record types |
|---|---:|---|
| [1753878](../results/cluster/1753878/) | 1 | cmwm_g0_attestation |
| [1753886](../results/cluster/1753886/) | 1 | supporting_metadata |
| [1758047](../results/cluster/1758047/) | 1 | cmwm_graph_isomorphism_gi0_audit |
| [1758049](../results/cluster/1758049/) | 1 | cmwm_graph_isomorphism_gi0_audit |
| [1758987](../results/cluster/1758987/) | 2 | cmwm_training_status, supporting_metadata |
| [1764055](../results/cluster/1764055/) | 26 | cmwm_chen_behavioral_readiness_v1, cmwm_engineering_training_run, cmwm_training_status, supporting_metadata |
| [1764060](../results/cluster/1764060/) | 1 | cmwm_chen_behavioral_readiness_v1 |
| [1764072](../results/cluster/1764072/) | 1 | cmwm_sherlock_graph_free_bm25_development_v1 |
| [1764075](../results/cluster/1764075/) | 1 | cmwm_sherlock_graph_free_bm25_development_v2 |
| [1766363](../results/cluster/1766363/) | 126 | cmwm_development_replication_summary_v1, cmwm_engineering_training_run, cmwm_training_status, supporting_metadata |
| [1771202](../results/cluster/1771202/) | 1 | cmwm_development_endpoint_coverage_audit_v1 |
| [1771216](../results/cluster/1771216/) | 1 | cmwm_validation_pool_endpoint_capacity_v1 |
| [1771259](../results/cluster/1771259/) | 1 | cmwm_validation_pool_endpoint_capacity_v1 |
| [1771270](../results/cluster/1771270/) | 1 | cmwm_development_endpoint_manifest_v1 |
| [1773008](../results/cluster/1773008/) | 1 | cmwm_e2_development_g0_readiness_v1 |
| [1773009](../results/cluster/1773009/) | 1 | cmwm_e2_development_g0_readiness_v1 |
| [1777162](../results/cluster/1777162/) | 1 | cmwm_g0_attestation |
| [1777163](../results/cluster/1777163/) | 1 | cmwm_development_endpoint_manifest_v1 |
| [1779882_e2_final](../results/cluster/1779882_e2_final/) | 125 | cmwm_endpoint_predictions_v1, cmwm_engineering_training_run, cmwm_training_status, supporting_metadata |
| [1806489_smoke](../results/cluster/1806489_smoke/) | 5 | cmwm_engineering_training_run, cmwm_training_status, supporting_metadata |
| [1806501_e3_final](../results/cluster/1806501_e3_final/) | 1 | cmwm_e3_development_summary_v1 |
| [e3_preflight_20260926](../results/cluster/e3_preflight_20260926/) | 2 | cmwm_development_endpoint_manifest_v1, cmwm_g0_attestation |

## Later Analysis of Saved Results

The [11 November sensitivity report](../results/analysis/saved_endpoint_sensitivity_2026-11-11/REPORT.md)
and [machine-readable values](../results/analysis/saved_endpoint_sensitivity_2026-11-11/sensitivity.json)
separate ordinary answers from source-record selection. This reuses existing E2/E3
outputs; it is not a new training run or an independent replication.

## Historical FilmFest Records

Behavioral reproduction and exploratory graph diagnostics are separate from the symbolic model study. Predicted-graph targets are not independent ground truth.

- [results/cluster/1752460/artifacts/behavioral_statistics.json](../archive/filmfest/results/cluster/1752460/artifacts/behavioral_statistics.json)
- [results/cluster/1752460/artifacts/reproduction_evaluation.json](../archive/filmfest/results/cluster/1752460/artifacts/reproduction_evaluation.json)
- [results/cluster/1752617/RUN_REPORT.md](../archive/filmfest/results/cluster/1752617/RUN_REPORT.md)
- [results/cluster/1752617/synthetic_smoke/gate_b.synthetic.json](../archive/filmfest/results/cluster/1752617/synthetic_smoke/gate_b.synthetic.json)
- [results/cluster/1752617/synthetic_smoke/synthetic_smoke_summary.json](../archive/filmfest/results/cluster/1752617/synthetic_smoke/synthetic_smoke_summary.json)
- [results/cluster/1753894/synthetic_smoke/gate_b.synthetic.json](../archive/filmfest/results/cluster/1753894/synthetic_smoke/gate_b.synthetic.json)
- [results/cluster/1753894/synthetic_smoke/synthetic_smoke_summary.json](../archive/filmfest/results/cluster/1753894/synthetic_smoke/synthetic_smoke_summary.json)
- [results/cluster/CLSP_RUN_REPORT.md](../archive/filmfest/results/cluster/CLSP_RUN_REPORT.md)
- [results/cluster/PREDICTED_GRAPH_EXPLORATORY_REPORT_2026-08-16.md](../archive/filmfest/results/cluster/PREDICTED_GRAPH_EXPLORATORY_REPORT_2026-08-16.md)
- [results/cluster/causal_graph_v2_1752621/RUN_REPORT.md](../archive/filmfest/results/cluster/causal_graph_v2_1752621/RUN_REPORT.md)
- [results/cluster/predicted_retrieval_v2_1753895/exploratory_diagnostics.json](../archive/filmfest/results/cluster/predicted_retrieval_v2_1753895/exploratory_diagnostics.json)
- [results/filmfest_reproduction/REPRODUCTION_REPORT_V1.md](../archive/filmfest/results/filmfest_reproduction/REPRODUCTION_REPORT_V1.md)
- [results/filmfest_reproduction/behavioral_statistics.json](../archive/filmfest/results/filmfest_reproduction/behavioral_statistics.json)
- [results/filmfest_reproduction/behavioral_statistics_v1.json](../archive/filmfest/results/filmfest_reproduction/behavioral_statistics_v1.json)
- [results/filmfest_reproduction/red_team/RED_TEAM_REPORT.md](../archive/filmfest/results/filmfest_reproduction/red_team/RED_TEAM_REPORT.md)
- [results/filmfest_reproduction/red_team/cluster_1752543/independent_validation.json](../archive/filmfest/results/filmfest_reproduction/red_team/cluster_1752543/independent_validation.json)
- [results/filmfest_reproduction/red_team/sensitivity_summary.json](../archive/filmfest/results/filmfest_reproduction/red_team/sensitivity_summary.json)
- [results/filmfest_reproduction/reproduction_evaluation.json](../archive/filmfest/results/filmfest_reproduction/reproduction_evaluation.json)
- [results/filmfest_reproduction/reproduction_evaluation_v1.json](../archive/filmfest/results/filmfest_reproduction/reproduction_evaluation_v1.json)

## Dated plans and reports

These describe their original dates, not current execution authorization.

- [ARCHITECTURE.md](../archive/notes/ARCHITECTURE.md)
- [BENCHMARK_RUBRIC_2026-10-12.md](../archive/notes/BENCHMARK_RUBRIC_2026-10-12.md)
- [CHEN_DATA_SCOPE.md](../archive/notes/CHEN_DATA_SCOPE.md)
- [DATASET.md](../archive/notes/DATASET.md)
- [DESIGN_AUDIT.md](../archive/notes/DESIGN_AUDIT.md)
- [E2_RESULTS_2026-09-01.md](../archive/notes/E2_RESULTS_2026-09-01.md)
- [E3_COMPLETION_CHECKLIST.md](../archive/notes/E3_COMPLETION_CHECKLIST.md)
- [E3_RESULTS_2026-09-28.md](../archive/notes/E3_RESULTS_2026-09-28.md)
- [EXECUTION_REPORT_2026-08-16.md](../archive/notes/EXECUTION_REPORT_2026-08-16.md)
- [EXECUTION_REPORT_2026-08-24.md](../archive/notes/EXECUTION_REPORT_2026-08-24.md)
- [EXECUTION_REPORT_2026-08-26.md](../archive/notes/EXECUTION_REPORT_2026-08-26.md)
- [EXECUTION_REPORT_2026-09-26.md](../archive/notes/EXECUTION_REPORT_2026-09-26.md)
- [EXPERIMENTS.md](../archive/notes/EXPERIMENTS.md)
- [GI1_PROTOCOL_DRAFT.md](../archive/notes/GI1_PROTOCOL_DRAFT.md)
- [GRAPH_ISOMORPHISM_STUDY.md](../archive/notes/GRAPH_ISOMORPHISM_STUDY.md)
- [NEXT_EXPERIMENTS_SETUP_2026-09-30.md](../archive/notes/NEXT_EXPERIMENTS_SETUP_2026-09-30.md)
- [OPENNEURO_METADATA_AUDIT.md](../archive/notes/OPENNEURO_METADATA_AUDIT.md)
- [PUBLICATION_ROADMAP.md](../archive/notes/PUBLICATION_ROADMAP.md)
- [RESEARCH_MAP.md](../archive/notes/RESEARCH_MAP.md)
- [SMOKE_TEST_PLAN.md](../archive/notes/SMOKE_TEST_PLAN.md)
- [TESTING.md](../archive/notes/TESTING.md)
