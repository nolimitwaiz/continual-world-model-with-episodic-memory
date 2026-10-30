> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# Testing and Experiment Boundaries

## What a test means here

Engineering tests establish that the implementation obeys its contract. They do not establish that
event memory works. Scientific experiments compare conditions and may support or reject the
hypothesis.

## Local-safe engineering checks

The local machine may run only tiny deterministic checks that cannot produce a research result:

- Python syntax/import checks;
- configuration-schema checks;
- action precondition and state-transition unit tests;
- deterministic seed and checksum tests on hand-sized fixtures;
- boundary indexing and tolerance-matching tests;
- factual/counterfactual causal-edge consistency tests;
- decision-gate arithmetic tests;
- the negative cluster guard, which must refuse training without Slurm and CUDA.

No local training loop, dataset-scale generation, inference benchmark, parameter tuning, profiling,
or model comparison is authorized.

## Cluster engineering checks

After explicit access and submission authorization, the cluster will run:

- dependency/environment verification;
- G0 dataset generation and validation;
- one-batch shape and numerical checks;
- deterministic replay on CUDA;
- checkpoint/resume round trips;
- resource profiling for the pilot budget.

The integrated trainer additionally writes named atomic checkpoints and `last.pt` at the declared
cadence. Checkpoints include model, optimizer, sampler RNG, Torch/CUDA RNG, metric history, elapsed
time, peak CUDA allocation, and code/config/G0 identities. Resume must fail closed on any identity
mismatch. The first G0-compatible trainer validation is 20 steps with evaluation and checkpointing
every five steps; it is an engineering run, not a pilot or scientific result.

The executable smoke suites and pass conditions are frozen in
SMOKE_TEST_PLAN.md (historical source location: `SMOKE_TEST_PLAN.md`). `python scripts/submit_smokes.py --dry-run` resolves the
ordered suite without invoking Slurm; a real invocation remains protected by scheduler arguments,
the authorization token, and `--confirm-submit`.

These runs are labeled `engineering` and cannot support a scientific claim.

## Scientific experiment sequence

1. Establish fixed-context, frame-memory, fixed-chunk, and matched-random baselines.
2. Run oracle event memory under the same budgets.
3. Stop if the oracle-value gate fails.
4. Train a causal online boundary detector only if the oracle passes.
5. Stop or revise once if the detector gate fails.
6. Render the sealed episodes and test one frozen DINOv3 interface only if learned boundaries pass.

Confirmatory comparisons use five paired seeds, identical manifests, a single primary endpoint,
paired confidence intervals, two held-out families, and one matched CUDA hardware class.

## Checks completed locally so far

- all Python source files pass bytecode compilation when the cache is redirected to a writable
  temporary path;
- deterministic symbolic generation and cross-split validation pass on a four-episode fixture;
- boundary segmentation and decision-gate arithmetic checks pass;
- eight hand-authored G0 audit fixtures pass for manifest tamper detection, leakage findings, query
  balance, actual FIFO eviction, full segment-length matching, generator-diversity failures,
  metadata shortcuts, and causal-confound failures;
- the training command correctly refuses execution outside Slurm.
- the cluster smoke command also correctly refuses execution outside Slurm;
- random-matched segmentation now preserves the complete reference segment-length multiset rather
  than matching boundary count alone;
- the new smoke and submission sources pass Python bytecode compilation, and all Slurm scripts pass
  `bash -n` syntax validation.

Source contracts have also been authored for symbolic state/action encoding, the fixed GRU,
key/value writing, query encoding, and pointer outputs. Their Python syntax compiles, but their
tensor/shape tests have not been executed locally because PyTorch is unavailable and model execution
is reserved for authorized cluster engineering checks.

The complete pytest suite has not been run because the available local Python environments do not
contain the declared development dependencies (`pytest`, `PyYAML`, and PyTorch). No dependency was
installed and no experiment was substituted for those tests.

## Submission consent boundary

Repository files under `cluster/` are inert templates. Creating or validating them does not contact
a cluster. A real submission requires all of the following:

- an explicitly authorized login/transfer method;
- the target hostname/environment;
- a valid Slurm allocation/account;
- partition and QoS choices;
- approved cluster storage paths;
- an explicit instruction to transfer and/or submit.

Until those are supplied and authorized, no `ssh`, `scp`, `rsync`, or `sbatch` action will be taken.
Even after scheduler settings exist, the submission helper refuses a real job unless both
`--confirm-submit` and the exact `CMWM_SUBMISSION_AUTHORIZED` token are supplied.

Once a training job is submitted, research workflow does not remain active to poll it continuously. The user
receives the Slurm ID and a conservative return time; checking the scheduler and artifacts happens
only after a later user message.
