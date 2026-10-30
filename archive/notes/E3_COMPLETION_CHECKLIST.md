> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# E3 completion and paper-evidence checklist

Scope: complete the existing D-039 matrix, not a new architecture search. A reliable null or
negative result is a valid outcome. The user requested persistent follow-through on 2026-09-26.

## Active execution

- GPU matrix: Slurm array 1806501, frozen 25 configurations, at most three A100 tasks at once.
- Run root: `<CLUSTER_PATH>`.
- Source: `<CLUSTER_PATH>` (do not edit during execution).
- Environment and immutable endpoint paths: `cluster/executions/e3_20260926/environment.sh`.
- Separate CPU results-auditor tests: job 1806516 COMPLETED, exit 0:0. Four tests passed, including
  a fabricated complete 25-run fixture, missing-run rejection, ten corruption subcases, the real
  smoke prediction artifact, and the correct primary-versus-secondary gate distinction.

## Completion criteria

1. All five conditions and five seeds complete 1,000 updates, without duplicate overlapping jobs.
2. Every final result matches the frozen config/source/data/endpoint; all runs use the same 512
   unique questions and eight equal task/delay cells.
3. All ten evaluation checkpoints per run have intact prediction artifacts and recomputable
   scores. Confirm checkpoint indexes/files, losses, gradients and resource records.
4. Report C4 minus C3 as the primary contrast. Always report C4 minus C2 and all five seed pairs.
5. Report D-039's practical-effect threshold, paired-seed interval, and sign consistency. Do not
   replace a failed primary result with a favorable secondary task. Cellwise intervals are descriptive.
6. Explain remaining limitations and what additional held-out evidence would be needed for a paper.
   Development completion does not automatically open C5, fMRI, fast weights, or the test split.

## Results analysis

`cluster/executions/e3_20260926/analyze_e3.py` is outside the frozen executable tree. It validates
128,000 saved prediction rows across the full learning curves before producing `summary.json` and
`REPORT.md`. It fails closed on missing runs, altered configs, mismatched identities, missing
checkpoints, corrupted predictions, mismatched scores, or nonfinite metrics. Checkpoint files are
indexed but their tensors are not all reloaded by this aggregate audit; the smoke validator does
reload its own four checkpoints. Do not overstate this distinction.

Run the analysis only after its CPU tests pass and the full matrix is complete, using a new output
directory. Do not use the obsolete E1 replication aggregator.

## Failure recovery

Inspect scheduler reason and stderr before retrying. For an execution-only timeout/failure, use
the exact resolved config and identity-checked last checkpoint; avoid restarting from step zero.
Never resubmit completed runs. Do not change model, precision, seeds, budgets, or endpoint to rescue
accuracy. A scientific or identity mismatch requires diagnosis and a recorded protocol decision.

## Follow-up status

An hourly thread follow-up was attempted but rejected by the app safety reviewer because of the
earlier user preference to stop after submission. No recurring automation was created. A concise
confirmation question has been presented to resolve that conflict. Do not bypass that rejection
with an external daemon, cron job, another task, or an indirect self-resubmitting controller.
The existing Slurm training jobs continue independently of this conversation.
