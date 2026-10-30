> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# CMWM-E1 Experimental Protocol

## 1. Confirmatory question

Under matched parameter, training-example, storage, retrieval, and tuning budgets, do oracle and learned event boundaries improve long-delay memory performance over the strongest non-event memory strategy?

This document separates engineering validation, exploration, and confirmatory testing. Only confirmatory runs can support a research claim.

## 2. Null hypotheses

### H0-O: oracle event value

Under the declared budgets, oracle-event memory does not improve the primary score over the strongest development-selected non-event condition among fixed context, frame retrieval, fixed chunks, and random matched chunks.

Formally, with primary-score difference `delta_oracle`:

```text
H0-O: delta_oracle <= 0
```

The practical gate is stricter than statistical rejection: the effect must also meet the minimum effect size and replication requirements in G1.

### H0-L: learned event value

A causal, online boundary detector does not preserve a practically useful fraction of the oracle-event benefit.

```text
H0-L: delta_learned / delta_oracle < 0.75
```

The learned detector must also meet the boundary-quality and direct-baseline requirements in G2.

### H0-V: visual transfer

Replacing oracle symbolic state with frozen visual features eliminates the learned-event advantage.

```text
H0-V: delta_visual <= 0
```

Prototype 1 contains no hypothesis about graph memory, alternative dynamics, recursive computation, test-time learning, or non-backpropagation learning.

Graph isomorphism is tracked separately in `GRAPH_ISOMORPHISM_STUDY.md` (historical source location: `./GRAPH_ISOMORPHISM_STUDY.md`).
GI-0 is a no-training structural audit and does not add graph memory to Prototype 1. A GI-1
retrieval comparison may be preregistered only after GI-0 passes its invariance, separation, and
held-out motif-coverage diagnostics. GI-0 passed on CLSP job 1758049; the permitted protocol-only
next step is recorded in `GI1_PROTOCOL_DRAFT.md` (historical source location: `./GI1_PROTOCOL_DRAFT.md`). GI-1 is not authorized
to run and does not alter the E1 sequence.

## 3. Dataset protocol

All data are generated on the cluster from a versioned manifest.

### Scale per condition and seed

- training: 10,000,000 observation steps;
- development: 10,000 episodes;
- confirmatory test: 20,000 episodes, split evenly across two held-out generator families;
- fixed context: 32 observations;
- evaluated query lags: 16, 32, 64, and 128 observations.

The same episode manifests and queries are used for every paired condition.

### Split design

Random episode splits are insufficient. The test set contains two disjoint families:

- **HF-A, role and persistence shift:** unseen combinations of entity appearance, role, containment, occlusion duration, and action order;
- **HF-B, causal composition shift:** unseen compositions and lengths of known causal mechanisms with structurally matched distractors.

No generator seed, rendered episode, symbolic trajectory, causal composition identifier, or query appears in more than one split. Test manifests are sealed before model development and are not used for architecture or hyperparameter selection.

### Generator validation gate G0

Before model comparison:

- a deterministic oracle evaluator must answer every generated query exactly;
- a current-context control must exceed 95% accuracy for lag-16 in-window queries;
- label-frequency, query-template, and sequence-length audits must show that outcome labels cannot be predicted from a single metadata field;
- causal-query construction must independently vary intervention-supported causal relevance,
  temporal distance, entity overlap, and semantic similarity, with recency-only, metadata-only, and
  target-field-only shortcut baselines reported;
- matched causal distractors must prevent the answer from being recovered from the disclosed
  reference step or a deterministic target-field-to-action mapping;
- deliberate future-state, event-ID, and causal-edge leakage tests must fail to find a path into model inputs;
- repeated generation with the same manifest must produce identical symbolic trajectories and checksums.
- at the primary lags, the referenced frame must fall outside the 32-step context and outside the
  retained 64-record FIFO frame window; occupancy/eviction audits must demonstrate actual storage
  pressure rather than infer it from configuration.

Failure blocks all scientific runs.

## 4. Conditions and ordering

Conditions are run in this order:

1. C0 fixed context;
2. C1 frame retrieval;
3. C2 fixed chunks;
4. C3 random length-matched chunks;
5. select the strongest non-event condition on development data;
6. C4 oracle event boundaries;
7. only after G1 passes, C5 learned event boundaries;
8. only after G2 passes, repeat C0 through C5 with one frozen DINOv3 interface shared by every
   condition.

Do not run learned event detection merely because its implementation exists. Do not run visual transfer merely because oracle events look qualitatively interesting.

## 5. Primary and secondary metrics

### Primary score

For each seed and held-out family, calculate exact-match accuracy for the four task families at lags 64 and 128. First macro-average across lags within a task, then across the four tasks, then across HF-A and HF-B. This result is the **Long-Delay Memory Score (LDMS)** in `[0, 1]`.

No other aggregation may replace LDMS after confirmatory evaluation begins.

### Minimum effect

For baseline error `e_b = 1 - LDMS_baseline` and candidate error `e_c = 1 - LDMS_candidate`:

```text
absolute gain = LDMS_candidate - LDMS_baseline
relative error reduction = (e_b - e_c) / e_b
```

The minimum practical effect is either:

- at least `0.05` absolute LDMS gain; or
- at least `0.20` relative error reduction.

### Secondary metrics

- exact-match accuracy by task, lag, generator family, and seed;
- boundary precision, recall, and F1 with a tolerance of plus or minus one observation;
- boundary area under the precision-recall curve;
- boundary-probability Brier score and expected calibration error;
- boundary rate and performance stratified by cue type, including sensory change versus latent
  context/action change;
- retrieval precision at 1 and 4;
- memory occupancy and bytes written per episode;
- training FLOPs and wall time;
- inference FLOPs, median latency, and p95 latency per query;
- peak accelerator memory;
- parameter count;
- calibration error for answer confidence;
- failure rates by occlusion, containment, distractor count, and causal-chain length.

Secondary metrics are diagnostic unless named by a gate. Families of secondary significance tests use Holm correction.

## 6. Equal-budget controls

The following are hard constraints, not reporting suggestions.

### Data

- exactly 10,000,000 training observations per seed and condition;
- identical train, development, and test manifests across paired conditions;
- identical query counts and lag distribution;
- no additional supervision for event conditions except the boundary labels used to train C5.

Boundary-label cost is reported separately and cannot be hidden inside synthetic-data generation.

### Parameters

Trainable parameter counts must be within 2% across compared conditions. If the boundary detector exceeds that range, add a parameter-matched non-event control that receives the same inputs but cannot influence segmentation or retrieval.

### Training compute

All conditions receive the same observation budget and the same maximum optimizer-update count. Before confirmatory runs, the single-seed cluster pilot profiles total FLOPs. Models must be resized or schedules adjusted until total training FLOPs are within 5% of the strongest non-event baseline. A condition that cannot satisfy this is reported as an unmatched exploratory result and cannot pass a Prototype 1 gate.

### Tuning

Each condition receives at most 12 development trials from the same declared search space and per-trial budget. Test data are unavailable during selection. The strongest non-event baseline is selected by development LDMS before C4 test results are revealed.

### Storage

- maximum of 64 fixed-width memory records (68,096 bytes under the declared layout);
- record key and value are each 128 float32-equivalent values;
- all metadata bytes count toward the limit;
- compressed representations are compared by total bytes, not record count;
- FIFO eviction is used everywhere;
- no hidden cache or uncounted raw frame store is allowed.

### Retrieval and inference

- at most four returned records per query;
- identical retriever scoring and query head across C1 through C5;
- candidate median latency and inference FLOPs must each be no more than 1.25 times those of the strongest non-event baseline;
- all comparisons use the same batch size, precision, software environment, and GPU class.

An accuracy gain that violates a hard budget does not pass a gate.

## 7. Statistical protocol

Confirmatory seeds are fixed as:

```text
11, 23, 47, 89, 131
```

Each condition uses the same seeds and episode manifests. Report seed-level results and their mean.
The fitted seed—not each of the thousands of test episodes—is the independent replication unit.
Calculate the primary paired 95% interval with a hierarchical bootstrap that first resamples seeds
and then resamples paired test episodes within each selected seed. Also report a paired seed-level
`t` interval as a sensitivity analysis. An episode-only interval is pseudo-replication and cannot
pass a gate. The lower hierarchical confidence bound for a claimed positive difference must be
greater than zero.

Do not replace a failed seed, exclude an outlier, modify an aggregation, or add a task after results are visible. Any deviation creates a new exploratory protocol and cannot rescue the original confirmatory result.

## 8. Exact decision gates

### G1: oracle events

C4 passes only if all conditions hold:

1. its LDMS beats the development-selected strongest non-event baseline;
2. the paired 95% confidence interval for the LDMS difference excludes zero;
3. it achieves at least 0.05 absolute gain or 0.20 relative error reduction;
4. the LDMS difference is positive separately on HF-A and HF-B;
5. the seed-level difference is positive for at least four of five seeds;
6. every data, parameter, training, storage, retrieval, and hardware control passes.

If G1 fails, stop event-detector and visual-transfer work. Permit one diagnostic analysis using existing artifacts; do not tune C4 against the test set.

### G2: learned events

C5 passes only if all conditions hold:

1. `delta_learned / delta_oracle >= 0.75` using LDMS effects against the same baseline;
2. boundary F1 is at least 0.80 on each held-out family;
3. its paired LDMS advantage over the strongest non-event baseline has a 95% confidence interval excluding zero;
4. its LDMS difference is positive separately on HF-A and HF-B;
5. the seed-level difference is positive for at least four of five seeds;
6. every equal-budget control passes.

If G2 fails, allow one bounded revision defined in a new decision record before accessing additional test episodes. A second failure stops learned-boundary development in Prototype 1.

### G3: frozen DINOv3 visual transfer

The frozen-DINOv3 stage passes only if all conditions hold:

1. the learned-event LDMS advantage has a paired 95% confidence interval excluding zero;
2. it retains at least 50% of the P1A learned-event uplift;
3. the difference is positive separately on HF-A and HF-B;
4. the seed-level difference is positive for at least four of five seeds;
5. all equal-budget controls remain satisfied.

Failure means the symbolic-state result does not support a vision claim. It does not authorize
encoder training or architecture search within E1. This gate tests transfer of the memory effect,
not DINOv3 superiority; every condition uses the same frozen checkpoint, readout, and projection.

## 9. Cluster-only run policy

No experiment or executable research validation runs locally. Static checks, configuration validation, and deterministic unit tests over tiny hand-authored fixtures may run locally; they cannot generate a result, tune a method, or exercise a training/evaluation loop.

### Cluster run classes

- **Engineering:** imports, one-batch execution, numerical checks, determinism checks, checkpoint round trips, and data-schema validation.
- **Pilot:** one seed used for memory, FLOP, latency, and resource estimation and for choosing the bounded development search space.
- **Exploratory:** explicitly labeled investigations that may inform a future protocol but cannot support the current confirmatory claim.
- **Confirmatory:** frozen code commit, environment, manifests, configuration, GPU class, seeds, and metrics.

Every run records code commit, environment lock hash, container identifier, data-manifest hash, GPU model, driver/runtime versions, scheduler job ID, seed, configuration, stdout/stderr, metrics, checkpoints, and resource utilization.

Timing and energy comparisons are valid only within a matched GPU class. Runs interrupted by infrastructure failure may be resumed from an auditable checkpoint; statistical failures may not be rerun selectively.

## 10. Stop conditions and report

The branch stops when a required gate fails, the equal-budget contract cannot be met, leakage is discovered, or confirmatory reproducibility fails.

The final Prototype 1 report must include:

- all confirmatory conditions, including negative results;
- seed-level and aggregate metrics;
- confidence intervals and effect sizes;
- budget-compliance table;
- failure and leakage audits;
- exploratory work clearly separated from confirmatory evidence;
- a keep, reject, or retest decision for event memory;
- no recommendation to scale unless G1, G2, and G3 all pass.
