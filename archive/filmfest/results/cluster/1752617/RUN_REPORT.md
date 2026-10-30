> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# CLSP synthetic benchmark smoke report

Date: 2026-08-12  
Cluster account: `<CLUSTER_ACCOUNT>`  
Remote snapshot: `<CLUSTER_PATH>`  
Successful Slurm job: `1752617`

## Outcome

The causal-retrieval benchmark plumbing passed an isolated, code-and-synthetic-data-only run on
CLSP. Six adversarial unit tests passed, all generated artifacts passed SHA-256 verification on
the cluster and after download, and the full synthetic pipeline completed in 0.46 seconds inside
a two-second Slurm job.

This is engineering validation only. No Chen Lab-derived table was uploaded for this run, no
FilmFest comparative model was executed, and the synthetic scores are not scientific findings.

## Resources

| Field | Value |
|---|---|
| Partition | `cpu` |
| Account | `default` |
| Node | `octopod` |
| Allocated CPUs | 4 |
| Requested memory | 4 GB |
| Peak batch RSS | 3,288 KB |
| Slurm elapsed | 2 s |
| Python | 3.10.12 |

An earlier job, `1752616`, failed in two seconds because the test intended to perturb only test
scores reused base-question IDs that are training examples in a second fold. The pipeline itself
did not fail. The test was corrected to use a single fold, preserving the intended leakage check,
and the retry was submitted under a new job ID rather than overwriting the failed record.

## Checks exercised

- Causal path construction excludes the anchor and is deterministic without semantic scores.
- Candidates are from the same story and strictly earlier than the anchor.
- Overlap thresholds are estimated from training rows only.
- Backward causal edges fail validation.
- Manifest validation, BM25 retrieval, oracle causal retrieval, aggregation, and SVG generation
  complete end to end.
- The Gate B evaluator passes a constructed positive-control case only when coverage and both
  prespecified effects pass.

The synthetic pipeline contained 16 events, 7 retained directed edges, 15 base questions, 30
fold-expanded manifest rows, 15 out-of-fold test questions, and 30 retrieval result rows. Its Gate
B result was correctly `inconclusive`: only 3 long/low questions across 2 stories were available,
below the frozen minimum of 20 questions across 3 stories, and no dense result was present.

## Scientific execution gate

The real FilmFest Gate B comparison remains blocked on the exact directed causal edge judgments
and event crosswalk. Published causal centralities cannot reconstruct paths and are not a valid
substitute. The guarded A100 template therefore fails closed before allocating model work while
the freeze preconditions are false.

Once those inputs arrive and validate, planned resource mapping is:

1. CPU: graph/path manifest, leakage validation, BM25, aggregation, bootstrap, and statistics.
2. One A100 per fold/task: MPNet overlap audit, E5 dense retrieval, and Qwen generation. The 7B
   generator fits on one A100; tensor-parallel multi-GPU execution is unnecessary.
3. A100 job array: independent Song EM-GRU seeds/folds can run concurrently, one GPU per task.

Current planning estimate after the graph inputs arrive is 15–30 minutes of CPU allocation and
approximately 2–6 aggregate A100-hours for the primary baseline suite. Queueing, first-time model
cache population, and faithful Song asset reconciliation could make wall time 4–12 hours or extend
the predecessor reproduction into the next day. These are estimates; exact question count and
faithful predecessor inputs must be known before a defensible schedule can be frozen.
