> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# CMWM Execution Report — 2026-08-26

## Completed seed-0 synthetic development pilot

Array `1764055` completed all five 1,000-update conditions on a shared 512-example development
cohort. It did not evaluate the test split and every result records `claim_eligible: false`.

| Condition | Final LDMS | Best LDMS | Final causal accuracy | Retrieval recall@4 |
| --- | ---: | ---: | ---: | ---: |
| C0 context | 0.380859 | 0.392578 | 0.000000 | N/A |
| C1 frames | 0.468750 | 0.482422 | 0.312757 | 0.294922 |
| C2 fixed | 0.832031 | 0.841797 | 0.925926 | 0.886719 |
| C3 random matched | 0.820312 | 0.820312 | 0.893004 | 0.945312 |
| C4 oracle events | **0.876953** | **0.876953** | **1.000000** | **0.978516** |

C4 exceeded C2 and C3 at all ten checkpoints. At step 1,000, C4-minus-C2 was 0.044922 and
C4-minus-C3 was 0.056641. C4's relative error reduction over C2 was 26.7%, above the declared 20%
practical threshold. The result is still one-seed development evidence and cannot pass G1.

## Completed Sherlock corrected development evaluation

Job `1764075` completed the versioned v2 BM25 evaluation on 401 queries from 12 development
participants. The five sealed participants were not scored.

- primary macro best-IoU hit@5: human 0.697688, random-matched mean 0.715328, difference -0.017640;
- macro best-IoU hit@1: human 0.474365, random 0.432085;
- mean reciprocal rank: human 0.590968, random 0.575219;
- macro IoU-nDCG@5: human 0.592478, random 0.584585.

The frozen development primary is null. Earlier-rank and graded-alignment diagnostics favor human
events and motivate, but do not validate, the fixed-width dense study.

## Confirmatory implementation gap

The current synthetic trainer evaluates a fixed 512-example validation cohort and calculates LDMS
as pooled state-plus-causal accuracy. The written G1 protocol instead requires an HF-A/HF-B test
endpoint macro-averaged by task, lag, and family, plus saved paired predictions and hierarchical
confidence intervals. The sealed G0 contract also contains 1,000 test episodes while the written
confirmatory reference names 20,000. Therefore no current result or submitted job is labeled
confirmatory.

## Five-seed development replication

- full unit gate: job `1764977`, 101 tests passed;
- array: `1764981`, 25 tasks = C0--C4 crossed with seeds 11/23/47/89/131;
- concurrency: at most eight ordinary GPUs, one GPU per task;
- scale: 1,000 updates, batch 32, fixed 512-example validation cohort per seed;
- test access: none;
- dependent aggregation: CPU job `1764982`, released only after every array task succeeds;
- expected duration: approximately 6--12 hours depending on queueing and the slower frame-memory
  condition.

The aggregation selects C2 from the independent seed-0 pilot, uses final-step paired effects, and
refuses missing/mismatched/claim-eligible artifacts. It explicitly records that G1 is not evaluated.

## General-GPU failure and A100 repair

Original array `1764981` completed six results: all five C0 seeds and C1 seed 11. The remaining 19
tasks were scheduled onto `c10` or `c04` and failed before optimization step 1. On `c10`,
`torch.cuda.is_available()` initially passed but model placement failed with “CUDA-capable devices
busy or unavailable.” On `c04`, the launcher exited because CUDA was unavailable. The resolved
configs were valid and no failed task wrote a checkpoint.

Completed C0 final LDMS values were 0.351563, 0.398438, 0.378906, 0.388672, and 0.365234 for seeds
11, 23, 47, 89, and 131 respectively: mean 0.376563. C1 seed 11 finished at 0.435547.

At the user's direction, compatibility job `1766360` tested the CUDA-12.1 environment on
`gpu-a100` node `e03`. It passed real BF16 forward, backward, allocation, and synchronization on an
80 GB A100-SXM4. Retry array `1766363` contains the exact 19 failed resolved configurations and
uses account `a100acct`; up to eight were requested, while account QoS currently permits three
simultaneous A100s. Replacement aggregator `1766367` has an `afterok` dependency on the retry.
Original aggregator `1764982` remains blocked by the failed original array.

## Completed five-seed aggregate

A100 retry array `1766363` completed all 19 tasks successfully. Replacement CPU job `1766367`
accepted all 25 result identities and paired validation cohorts.

| Condition | Mean final LDMS | Mean state accuracy | Mean causal accuracy | Mean retrieval@4 |
| --- | ---: | ---: | ---: | ---: |
| C0 context | 0.376563 | 0.699111 | 0.000000 | N/A |
| C1 frames | 0.478516 | 0.629199 | 0.301075 | 0.290625 |
| C2 fixed | 0.841016 | **0.753516** | 0.943167 | 0.901172 |
| C3 random matched | 0.803516 | 0.713057 | 0.908808 | 0.949219 |
| C4 oracle events | **0.850781** | 0.730205 | **0.992219** | **0.981250** |

C4 versus C3 was robust on the development proxy: mean +0.047266, paired 95% t interval
0.036502--0.058029, all five seed differences positive, and 24.1% relative error reduction.

C4 versus the independently selected C2 baseline did not pass the D-032 development gate: mean
+0.009766, interval -0.015025--0.034557, only two positive seeds plus one tie, and 6.1% relative
error reduction. The result is a null for the frozen pooled development primary and C5 remains
closed.

The secondary results reveal a task tradeoff rather than uniform equivalence: oracle events improve
causal retrieval and target availability, whereas fixed chunks improve state accuracy. The written
protocol always required task/lag macro-averaging, but the current trainer implements pooled
accuracy. The next permitted action is therefore a validation-only evaluator-conformance repair
using saved paired predictions; it is not test access or a post-hoc endpoint substitution.

## Evaluator-conformance coverage audit

On 2026-08-29, CPU-only CLSP job `1771202` completed before checkpoint evaluation. It passed five
new evaluator tests, regenerated each exact five-seed validation cohort, verified its stored SHA-256
identity across all conditions, and audited the eight required task-by-lag cells. The audit corrects
metadata only: causal lag is measured as ask step minus the hidden oracle source-action step, while
the model input continues to conceal that step. It performed no training, did not load model
weights, and did not access the test split.

The audit failed closed because no seed has all eight required cells. Delayed-state@128 is absent
for all five seeds; occlusion@64/128 is absent except for two single examples; causal@64/128 ranges
from zero to four examples per cell; and each 512-row cohort repeats 20--33 query IDs. Therefore the
protocol endpoint cannot be computed from the frozen sampled cohorts. The checksummed artifact is
archived at `results/cluster/1771202`.

CPU-only job `1771216` then scanned all 1,000 validation episodes without model predictions. Before
seeing its counts, the capacity target was fixed at 64 unique examples in each of the eight cells
(512 total). The pool contained 7,711 unique queries and no duplicate IDs, but exact required counts
were delayed-state@128 = 0, occlusion@128 = 0, occlusion@64 = 4, causal@64 = 36, causal@128 = 38,
delayed-state@64 = 152, location-history@64 = 524, and location-history@128 = 781. The checksummed
artifact is archived at `results/cluster/1771216`.

The existing E1 pool therefore cannot support the endpoint at any sampling policy. D-035 records
CMWM-Synth-E2 as the forward path: deliberately construct exact-lag task instances, freeze balanced
unique manifests, rerun G0, and retrain unchanged C0--C4 conditions. E1 checkpoints remain immutable
diagnostics and cannot be reused as E2 evidence. Test data, C5, and new architecture components
remain closed.

## E2 construction repair

The E2 generator now assigns an exact hidden-state anchor at lag 64 or 128, returns the object to
its origin after the anchor, and uses three trajectory variants. This yields nontrivial contained-
state, location-history, occlusion, and hidden-source causal queries at the same exact anchor without
changing model-visible query fields. The lag audit was also repaired so causal delay is computed
from its oracle source action in G0 metadata while the model continues to receive no source step.

CPU job `1771250` passed the full 108-test suite. Corrected capacity job `1771259` scanned the 1,000
E2 validation episodes and found exact-cell counts of 506/509 causal, 551/499 delayed state,
941/756 location history, and 501/499 occlusion for lags 64/128 respectively. All 10,923 query IDs
were unique, so the predeclared 64-per-cell gate passed. An earlier job, `1771255`, used a stale E1
audit launcher due to a staging omission; it reproduced E1 counts and is not scientific evidence.

CPU job `1771270` passed three manifest-construction tests and froze exactly 64 unique examples
from each of the eight E2 cells by fixed SHA-256 rank. The 512-query condition-blind development
manifest is `ff9188cff7f654232e3c24bcf8e1577bfd15d6afd8c6d13ac64aa17823cfb6df`; its seal is
`4db11ff48101dba82cd2b2bb5d42f930465685195623c195553946691f3606b4`; and its selected-query
identity is `b78bc3a018e30d4bbbd26f884e2dcfe7bf06e354e058ee44df08fcb7124a1603`. The checksummed
artifact is archived at `results/cluster/1771270`. It performed no model evaluation and no test
access.

Full CPU-only readiness job `1771291` was then submitted over all 10,000 E2 training and 1,000
validation episodes. It recomputes queries/answers, matched causal controls, oracle-versus-random
segment distributions, storage pressure, label balance, shortcut diagnostics, and development
manifest integrity. The script constructs only train and validation splits, records
`test_accessed=false`, and performs no optimization. A passing report is necessary but not by
itself sufficient to start training; any findings must first be reviewed and the final G0
attestation frozen.

Job `1771291` reached its one-hour wall limit without emitting an atomic report. It used only about
2.8 GB RSS and wrote no stderr, so this was not a data failure or memory exhaustion. Code inspection
identified repeated full-label Counter subtraction and majority scans for every query in the
leave-one-episode-out shortcut audit; unique causal source labels made that path near-quadratic.
The audit now computes the mathematically identical majority using cached per-episode exclusions
and globally ranked counters. An exhaustive small-counter test compares every optimized result to
brute-force subtraction.

The fail-closed rerun pipeline is `1773007` (full unit suite), `1773008` (1,000-train/100-validation
readiness audit with nonzero exit on scientific failure), and `1773009` (full 10,000/1,000 audit,
four-hour ceiling, released only after both predecessors pass). No job accesses test or trains a
model. Because the dataset-source identity includes audit code, the earlier `1771270` endpoint seal
must be regenerated after the G0 implementation is stable; it remains valid evidence that balanced
construction is possible, but it is not the final training authorization.

## Passing optimized E2 G0 audit

The optimized pipeline completed successfully. Unit job `1773007` passed 112 tests. Fail-closed
10% job `1773008` audited 1,100 episodes and 12,067 queries with zero errors in 2m43s. Full job
`1773009` audited 10,000 training plus 1,000 validation episodes and 120,443 queries with zero errors
in 27m07s. It recorded no test access, no flagged shortcut diagnostic, nondegenerate labels, exact
oracle/random-matched segment histograms, and 52,682/52,682 primary references both outside the
32-step context and evicted from the equal-byte 64-record frame FIFO. The only finding was the
pre-existing non-blocking warning that collision mechanics are not evidenced.

D-036 now makes test isolation structural: the E2 development config declares zero test episodes.
Jobs `1777161` (final unit suite), `1777162` (full trainer-compatible development G0 attestation),
and `1777163` (balanced endpoint re-seal under the final config/source identity) are dependency-
chained. No GPU training or test evaluation is part of this pipeline.

All three finalization jobs completed. The trainer accepted G0 artifact `1777162`; its seal digest,
dataset-source digest, 10,000/1,000/0 contract, and `test_accessed=false` record all verified. The
regenerated 512-query endpoint manifest from `1777163` is
`8b4fd22bba5f75904e825681c5f835028f24bb926ed31b28931795f982649a74`, with seal
`c60e9eab90c91f062c3b84422114b12342a453bc5acc7ba35186689ff79f72c0`. Checksummed local copies
are under `results/cluster/1777161`, `1777162`, and `1777163`.

D-037 submitted ordinary-GPU job `1777535`: one C4/oracle event-memory engineering smoke, seed 0,
20 updates, batch 32, checkpoint/evaluation every five updates, claim eligibility false, and zero
test episodes. Passing this job does not authorize the 25-run matrix; the balanced manifest and
saved per-query evaluator must first be integrated into the scientific training path.

Ordinary-GPU smoke `1777535` was assigned to `c04` and failed in six seconds before the trainer
started. PyTorch reported a CUDA initialization error and zero visible devices; the launcher refused
execution and no run artifact or checkpoint was written. The identical config/G0/source contract
was resubmitted as A100 job `1777544` on node `e02`, where CUDA compatibility had already been
validated. The retry is engineering-only, and hardware efficiency comparisons are excluded.
