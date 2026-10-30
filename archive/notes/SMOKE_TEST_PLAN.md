> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# CMWM-E1 Engineering Smoke-Test Plan

Status: cluster engineering validation only. Passing these tests does not establish that event
memory works.

## Principles

- Local execution is limited to syntax, configuration, unit, and fail-closed guard checks.
- Dataset generation, tensor execution, backward passes, checkpoints, profiling, and all model
  comparisons run inside an authorized Slurm CUDA allocation.
- Smoke tests use tiny data and at most two optimizer steps. Their metrics are discarded as
  scientific evidence.
- Learned-event training and frozen-vision work remain gate-locked even if their modules compile.
- Every smoke artifact records source-tree hash, resolved config hash, environment, GPU, Slurm job,
  seed, stdout/stderr, and pass/fail assertions.

## Local-safe preflight

| Test | Purpose | Pass condition |
| --- | --- | --- |
| L0 syntax | Import-independent Python bytecode compilation | Every source and test file compiles |
| L1 config matrix | Resolve and validate every YAML without execution | All names, dimensions, budgets, and run classes validate |
| L2 deterministic fixtures | Mechanics, query oracle, boundary indexing, manifest seal, and audit fixtures | All unit checks pass |
| L3 negative cluster guard | Ensure local training cannot start | Command exits with the declared refusal code |
| L4 submission dry run | Resolve the intended job matrix without writing remotely or invoking Slurm | Exact configs and scheduler placeholders are displayed; no `sbatch` call |

## Cluster engineering smoke matrix

| ID | Config/path | What it exercises | Required assertions |
| --- | --- | --- | --- |
| S0 environment | environment-only job | Python, package imports, CUDA, deterministic kernels, writable data/run roots | Correct versions logged; CUDA visible; write/read round trip succeeds |
| S1 generator replay | G0 tiny manifest | Deterministic generation, serialization, query recomputation, manifest seal | Two generations have identical hashes; every oracle answer recomputes |
| S2 G0 adversarial audit | G0 tiny plus deliberate corruptions | Leakage and shortcut detectors | Clean fixture has expected status; every injected fault is detected |
| S3 context model | `context_only` smoke overlay | Collation, state/action adapter, GRU, query head, loss, backward | Finite tensors/loss/gradients; shape contracts hold |
| S4 frame memory | `frame_memory` smoke overlay | Per-step writes, FIFO pressure, top-k retrieval | Exactly declared bytes; capacity and top-k never exceeded |
| S5 fixed chunks | `event_fixed` smoke overlay | Shared segment writer with fixed grouping | Same writer/head; segment coverage exact; no duplicated/dropped step |
| S6 random matched | `event_random_matched` smoke overlay | Full oracle segment-length permutation null | Per-episode length histogram matches oracle exactly |
| S7 oracle events | `event_oracle` smoke overlay | Oracle grouping without leaking boundary source to model | Only segment indices differ; tensor inputs contain no event/causal labels |
| S8 checkpoint | strongest implemented path | Save/load optimizer, model, RNG, config, and manifest IDs | Resumed next step matches uninterrupted step within tolerance |
| S9 determinism | repeat S3 and S7 | End-to-end reproducibility | Hashes/losses match under deterministic settings |
| S10 failure injection | any memory path | NaN, oversized record, missing candidate, corrupted seal | Job fails closed with a specific error |
| S11 resource pilot | after S0-S10 | Provisional memory/FLOP/latency envelope | Resource artifact produced; not compared scientifically |

## Scientific work that smoke tests cannot authorize

- choosing a winning memory condition from smoke losses;
- tuning event thresholds or hyperparameters;
- running learned boundaries before G1;
- running DINO before G2;
- claiming continual learning, graph reasoning, reinstatement, or credit assignment;
- pooling timing results across GPU classes.

## Initial cluster submission order

Submit serially at first:

1. S0 environment;
2. S1-S2 generator and audit;
3. stop if G0 readiness fails;
4. S3 context model;
5. S4-S7 memory code paths;
6. S8-S10 reproducibility and failure handling;
7. S11 resource pilot.

Do not submit an array until S0 succeeds and the single-job log location is verified. Do not submit
scientific C0-C4 runs until G0 passes on the intended dataset generator.

## Information required before transfer/submission

- exact SSH hostname or configured alias;
- username/authentication method (credentials remain outside the repository and logs);
- remote project directory;
- data, run, and model-cache directories;
- Slurm account/allocation, GPU partition, QoS, GPU type or constraint, and job-time ceiling;
- approved environment method (module/conda/venv/container) and Python/CUDA expectations.

The user's instruction to move to CLSP authorizes preparation and intended transfer/submission, but
the missing destination and scheduler values must be resolved before contacting the cluster.
