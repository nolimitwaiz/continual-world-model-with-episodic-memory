> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# E3 execution report — 2026-09-26

## Purpose and scope

Test whether event-aligned records improve long-delay recall under confusable object/event bindings.
This is the D-039 development experiment, not a completed world model or a confirmatory paper result.
The earlier E2 null result remains immutable. E3 is a post-E2 benchmark revision, and that chronology
must be disclosed in a paper. No training is run locally.

## Verified completed preparation

| CLSP job | Work | Verified outcome |
| --- | --- | --- |
| 1784622 | Frozen E3 unit suite | Completed; 115 tests passed |
| 1784623 | Full train/validation G0 audit | Completed in 43m49s; 11,000 episodes, 134,409 queries, zero errors |
| 1784624 | Condition-blind evaluation manifest | Completed; 512 unique questions, 64 per task/lag cell |

The audit explicitly does not establish collision mechanics. Test episodes and test access are zero.
Both returned JSON artifacts were copied into `results/cluster/e3_preflight_20260926/`.
The frozen remote checkout passes the trainer's seal and source-identity checks for all five configs.

## Submitted

- Job **1806489**: 20-update C4 engineering smoke, one A100, four CPUs, 16 GB host memory,
  30-minute wall-time ceiling. Checkpoint and evaluate at steps 5, 10, 15, and 20.
- Follow-up scheduler check: **COMPLETED, exit 0:0, on e02 in 6m29s**. The trainer records
  344.4 seconds of measured execution and four checkpoints. Result/config/status/index/curve JSON
  files were copied to `results/cluster/1806489_smoke/`. Final smoke LDMS is 0.271484375; this is
  an engineering diagnostic and is not a comparison or scientific result.
- The initial `default`/`gpu-a100` attempt was rejected before any job started. Allocation lookup
  confirmed the already-authorized `a100acct`/`base_qos` combination, used for the accepted job.
- Frozen code: `<CLUSTER_PATH>`.
- New output: `<CLUSTER_PATH>`.
- Launch records: `cluster/executions/e3_20260926/` locally and
  `<CLUSTER_PATH>` remotely.
- Environment: `<CLUSTER_PATH>` (Torch 2.5.1+cu121).

## Smoke gate passed; matrix submitted on follow-up

Read-only smoke validation passed: finite loss/gradients and model/optimizer tensors, four readable
checkpoints, matching source/config/G0/endpoint identities, and four intact 512-row paired prediction
artifacts with recomputed scores. See `results/cluster/1806489_smoke/VERIFICATION.md`. No hyperparameter
or seed selection used the smoke result.

Submitted **Slurm array 1806501**: unchanged five conditions (context, frames, fixed chunks,
random-matched chunks, oracle events) times seeds 11, 23, 47, 89, 131: **25 runs, 1,000 updates each**.
At most three simultaneous single-GPU A100 runs, four CPUs and 16 GB host memory per task, four-hour
wall-time ceilings, saved evaluation/checkpoints every 100 updates. Training remains cluster-only.

Array indices 0–4 are C0, 5–9 C1, 10–14 C2, 15–19 C3, 20–24 C4; within each condition the seed
order is 11, 23, 47, 89, 131. No matrix performance result is available at submission.

Post-submission verification: tasks 0, 1, and 2 are RUNNING on e02/e03; tasks 3–24 are PENDING
solely on `JobArrayTaskLimit` (the intended three-GPU concurrency cap). Submit time:
2026-09-26 08:01:16 UTC. Resolved manifest:
`<CLUSTER_PATH>`.
Suggested first check-in: about three hours after submission. Rough full-matrix planning estimate:
8–16 hours plus queue delays, based on earlier E2 runs rather than measured E3 throughput; revise
after the first completed E3 tasks. No completion promise or persistent monitoring is implied.

At the next check-in, inspect array state and artifacts before any retry. Existing completed runs
must not be repeated. The old `aggregate_development_replication.py` is E1-specific (directory names
and protocol) and is not a valid E3 report without explicit adaptation and validation.

Primary contrast: oracle minus random-matched. Report oracle minus fixed regardless of sign.
D-039's development pass requires at least +0.05 absolute or 20% relative error reduction, a paired
95% interval excluding zero, and at least four of five positive seeds. Passing this development
screen is not itself a confirmatory publication claim and does not authorize opening the test split.

Subsequent paper work depends on the outcome: independently held-out confirmation, pressure/delay/
interference controls, and a separately frozen human-data experiment. No broad superiority over RAG,
human-like cognition, fast weights, or complete multimodal world modeling follows from this matrix.

## User handoff policy

Report job IDs and a practical check-in interval; do not keep research workflow active throughout training.
No recurring monitor is installed. Unrelated cluster jobs are left untouched.

## Further authorized work after the persistence request

- Training status: three active runs each saved step 200; 22 remaining tasks await the three-task cap.
- Implemented `analyze_e3.py` and its tests outside the frozen executable tree. It validates the
  complete matrix and computes the prespecified comparisons, seed intervals and learning curves.
- CPU test job **1806516** completed with exit 0:0; all four tests passed in 8.027 seconds,
  including a full fabricated matrix fixture, missing-run rejection and ten corruption subcases.
  These tests are engineering evidence, not new scientific results.
- Hourly follow-up creation was blocked by app safety review; no automation exists. User confirmation
  was requested to explicitly supersede the earlier stop-after-submission preference.
