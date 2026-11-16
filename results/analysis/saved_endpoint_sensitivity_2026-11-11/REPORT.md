# Saved endpoint sensitivity analysis

11 November 2026. Exploratory reanalysis; no new training or test access.

The original composite endpoint remains primary. Answer-only scores exclude causal-pointer retrieval because its target space depends on grouping; this is a post-hoc diagnostic, not a new confirmatory endpoint.

## E2 scores

| Condition | Original composite | Answer tasks only | Pointer task only |
|---|---:|---:|---:|
| context_only | 64.96% | 86.61% | 0.00% |
| frame_memory | 63.91% | 85.21% | 0.00% |
| fixed_chunks | 90.12% | 86.82% | 100.00% |
| random_matched | 88.87% | 86.20% | 96.88% |
| oracle_events | 90.23% | 86.98% | 100.00% |

| Diagnostic | Contrast | Mean difference in points | Paired seed 95% interval | Positive seeds | Leave one seed out range in points |
|---|---|---:|---|---:|---|
| original_composite | oracle_minus_fixed | +0.12 | [-1.81, 2.05] | 2/5 | [-0.49, 0.59] |
| original_composite | oracle_minus_random | +1.37 | [-0.36, 3.09] | 4/5 | [0.83, 1.76] |
| answer_tasks_only | oracle_minus_fixed | +0.16 | [-2.42, 2.73] | 2/5 | [-0.65, 0.78] |
| answer_tasks_only | oracle_minus_random | +0.78 | [-0.66, 2.23] | 4/5 | [0.33, 1.11] |
| pointer_task_only | oracle_minus_fixed | +0.00 | [0.00, 0.00] | 0/5 | [0.00, 0.00] |
| pointer_task_only | oracle_minus_random | +3.12 | [0.56, 5.69] | 5/5 | [2.34, 3.71] |

## E3 scores

| Condition | Original composite | Answer tasks only | Pointer task only |
|---|---:|---:|---:|
| context_only | 33.01% | 44.01% | 0.00% |
| frame_memory | 31.37% | 41.82% | 0.00% |
| fixed_chunks | 56.29% | 42.55% | 97.50% |
| random_matched | 55.74% | 43.39% | 92.81% |
| oracle_events | 73.55% | 66.98% | 93.28% |

| Diagnostic | Contrast | Mean difference in points | Paired seed 95% interval | Positive seeds | Leave one seed out range in points |
|---|---|---:|---|---:|---|
| original_composite | oracle_minus_fixed | +17.27 | [2.15, 32.38] | 4/5 | [15.38, 22.56] |
| original_composite | oracle_minus_random | +17.81 | [3.60, 32.03] | 4/5 | [15.53, 22.71] |
| answer_tasks_only | oracle_minus_fixed | +24.43 | [4.59, 44.27] | 4/5 | [21.88, 31.38] |
| answer_tasks_only | oracle_minus_random | +23.59 | [4.91, 42.28] | 4/5 | [20.44, 30.01] |
| pointer_task_only | oracle_minus_fixed | -4.22 | [-5.98, -2.46] | 0/5 | [-4.69, -3.91] |
| pointer_task_only | oracle_minus_random | +0.47 | [-1.76, 2.70] | 2/5 | [-0.20, 0.78] |

## Data checks

E2: 25 runs, 12,800 endpoint prediction rows, 512 unique questions from 414 unique episodes. Every run has eight task/lag cells of 64 rows. Identity, correctness and target-eligibility checks passed.

E3 task/seed means reconstruct the saved composite. Its original per-query outputs are not available here, so no E3 query-level leakage, retention or episode-resampling claim follows.

Full task cells, seed differences, leave-one-out values, E2 target eligibility and source hashes are in `sensitivity.json`.

## Interpretation limits

- Post-hoc development sensitivity analysis, not independent replication or a new training experiment.
- Intervals describe paired training seeds on a reused cohort, not independent-world uncertainty.
- Task and leave-one-out analyses are exploratory; no multiple-comparison-adjusted claim is made.
- Pointer target eligibility is not a measured source-retention audit.
- Original E3 per-query outputs and checkpoint tensors are not included locally.
