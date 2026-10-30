> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# CMWM Execution Report — 2026-08-24

## 2026-08-25 continuation

The CUDA-12.1 general-GPU environment, exact-resume gate, corrected C0--C4 engineering matrix,
and the next development-pilot submission have now completed or advanced beyond the state
recorded below.

### Exact resume

- uninterrupted reference: job `1761784`, completed;
- intentional checkpoint-aligned interruption: job `1761785`, expected exit `75`;
- resumed run: job `1761786`, completed;
- exact comparison: job `1761787`, completed;
- result: model, optimizer, sampler RNG, Torch RNG, CUDA RNG, loss, validation metrics, best step,
  and best score were exactly equal.

### Corrected 20-step C0--C4 matrix

Jobs `1761795`--`1761799` completed on matched RTX 2080 Ti hardware with identical source,
dataset, G0, validation-cohort, and seed identities. Every condition saved checkpoints at steps
5, 10, 15, and 20.

| Condition | Final LDMS | Causal accuracy | Retrieval recall@4 |
| --- | ---: | ---: | ---: |
| C0 context | 0.015625 | 0.000000 | N/A |
| C1 frame | 0.078125 | 0.117647 | 0.171875 |
| C2 fixed chunks | 0.093750 | 0.176471 | 0.484375 |
| C3 random matched | 0.265625 | 0.441176 | 0.718750 |
| C4 oracle events | 0.312500 | 0.529412 | 0.843750 |

This ordered signal is engineering evidence only: one seed, 20 updates, and 64 validation
examples. C4 exceeded C3 by 0.046875 LDMS, below the provisional 0.05 absolute gate and below the
alternative 20% relative-error-reduction gate. C5 remains closed.

### 1,000-step development pilot

- cluster unit gate: job `1764054`, completed, **84 tests passed**;
- GPU array: job `1764055`, five conditions, one GPU each;
- scale: 1,000 updates, batch 32, fixed 512-example validation cohort, evaluation/checkpoint every
  100 steps, seed 0, no sealed-test evaluation;
- placement at verified start: tasks 0--2 on `c24` RTX 2080 Ti and tasks 3--4 on `c18` GTX
  1080 Ti. This is acceptable for a development convergence/effect-direction calibration, but
  cross-condition wall-time and hardware-efficiency comparisons are ineligible;
- evidence class: development only; `claim_eligible: false`.

### Chen behavioral readiness

The prior CLSP FilmFest/Sherlock workspace was rediscovered and retained rather than duplicated.
It includes 202 normalized FilmFest events, a 3,030-row recall matrix, 50 Sherlock scenes, 616
recall occurrences, an 850-row participant-scene matrix, and an exploratory model-predicted
FilmFest graph. CPU audit job `1764060` completed with all 11 readiness checks passing. It explicitly
records that the two graph-free behavioral datasets are ready, while no independent human directed
graph or raw fMRI is present. The returned artifact is preserved under
`results/cluster/1764060/readiness.json`.

### Sherlock graph-free development benchmark

- frozen protocol: the same 1,000 released encoding units grouped as fine units, fixed 20-unit
  chunks, 10 exact-length-matched random segmentations, or 50 released human scene groups;
- queries: timestamp-aligned participant recall text aggregated per coded subject-scene, with the
  scene label hidden from retrieval and used only for evaluation;
- split: 12 development participants and five sealed-test participants, frozen by SHA-256 rank
  before retrieval; only development is authorized;
- primary metric: participant-macro hit@5 with common BM25 retrieval; MRR, micro hit rates,
  returned tokens, and number of relevant records are diagnostics;
- test gate: job `1764071`, **89 tests passed**;
- first execution: job `1764070` failed closed before scoring because one released atomic row has
  equal start and end timestamps;
- correction: retain that row, audit exactly one zero-duration row and one part-boundary timestamp
  reset, and continue to group strictly by released row order;
- retry: job `1764072` completed on 401 development queries from 12 participants. Original
  participant-macro hit@5 was 0.697688 for human events and 0.839209 for the 10-seed
  random-matched mean;
- interpretation: the original overlap target is biased toward boundary-crossing records. Random
  chunks averaged 2.061 qualifying targets per query while human events always had one. The result
  is retained as an evaluation diagnostic, not evidence against event organization;
- versioned correction: v2 selects exactly one maximum-segment-IoU target record and separately
  reports graded IoU-nDCG. Job `1764074` passed 89 tests and job `1764075` was submitted. Because
  the correction followed v1 inspection, v2 also remains development-only;
- evidence boundary: development lexical baselines only. Record lengths and returned-token budgets
  are not matched, so this cannot establish superiority over all RAG or dense retrieval.

## Outcome

The checkpointed C0 trainer completed once on A100 hardware, an evaluator defect was discovered and
repaired, the expanded 79-test suite passed, and the first ordinary-GPU resume attempt correctly
failed closed because of an incompatible CUDA software environment. No scientific condition
comparison has run.

## Engineering jobs

| Job | Purpose | State | Evidence boundary |
| --- | --- | --- | --- |
| 1758987 | first checkpointed C0 BF16 run | failed before step 1 | exposed unsafe `-1e9` half mask |
| 1759587 | regression/unit suite after mask repair | 75 passed | engineering |
| 1759588 | 20-step C0 retry on A100 | completed; four checkpoints | execution only |
| 1759591 | fixed-cohort/resume-control unit suite | 79 passed | engineering |
| 1761532 | uninterrupted resume reference on c04 | failed at CUDA guard | no training |
| 1761533 | intentional-interruption path on c04 | failed at CUDA guard | no training |
| 1761534 | dependent resume path on c04 | failed at CUDA guard | no training |
| 1761535 | checkpoint comparison | dependency blocked | no comparison |

## Evaluator correction

Job 1759588 evaluated checkpoints 5, 10, 15, and 20 with seeds derived from the checkpoint step.
The 64 examples therefore changed between evaluations. The loss/checkpoint pipeline remains valid,
but its apparent best validation step and model-quality scores are excluded from all research
claims.

The repaired trainer constructs one deterministic validation cohort, records its SHA-256 identity,
and reuses it at every checkpoint. Metrics that are not defined for a condition are recorded as
unavailable. These controls passed job 1759591.

## Ordinary-GPU failure diagnosis

General node `c04` advertises four GPUs and was correctly allocated. Its NVIDIA driver reported
version 12.2 compatibility. The locked CMWM environment contains PyTorch `2.13.0+cu130`; CUDA 13.0
cannot initialize against that older driver, so the fail-closed trainer exited with code 3 before
model construction.

The remediation is a separate environment using an official CUDA 12.1 PyTorch wheel. The existing
CUDA-13/A100 environment remains immutable. Official installation commands are recorded by
[PyTorch](https://pytorch.org/get-started/previous-versions/).

## Next gate

1. Create and freeze the general-GPU environment.
2. Record Python, PyTorch, CUDA, driver, and GPU identities in a compatibility probe.
3. Pass the full unit suite.
4. Rerun uninterrupted, interrupted, and resumed C0 on one matched node.
5. Require exact equality of model, optimizer, sampler, Torch RNG, CUDA RNG, losses, and fixed-cohort
   metrics.
6. Only then run the corrected C0--C4 engineering matrix.
