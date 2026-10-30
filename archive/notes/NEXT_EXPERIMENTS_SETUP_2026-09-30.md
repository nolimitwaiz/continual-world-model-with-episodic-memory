> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# Next experiments: generalization and human-recall retrieval

**2026-10-12 priority update:** this draft is preserved, not the active launch queue. The user
requested small rubric-led speech/vision validation and no text-encoding expansion for now.
See `BENCHMARK_RUBRIC_2026-10-12.md` and D-041. Do not execute the older preparation inventory.

Status: local preparation only, 2026-09-30. No training, scientific evaluation, data generation,
dataset download, cluster connection, transfer, test access or submission is authorized by this
setup. Existing E2, E3 and Sherlock results remain unchanged. This is not a frozen preregistration.

## Central question

Can a bounded world-model memory preserve useful past information better by keeping meaningful
events together, rather than cutting experience into arbitrary pieces? The larger aim is a
world model that can maintain state and eventually predict changes using relevant past experience.
Current evidence concerns symbolic memory questions, not a complete predictive world model.

## A. Next main experiment: fresh-world confirmation

Evaluate the 25 existing final E3 checkpoints (five conditions by seeds 11/23/47/89/131) on a new,
separately audited and sealed held-out cohort. Do not select checkpoints by test performance, drop
seed 89, continue training, or tune the boundary detector. Reusing fitted checkpoints makes this
an independent-data confirmation, not a fresh training-seed replication.

Proposed endpoint: two held-out families, four tasks, lags 64/128, 64 examples per cell: 1,024 unique
questions from 256 episodes. Each episode supplies all four tasks at one assigned lag. Confirm
that this restriction is scientifically feasible before freezing the protocol; never silently
substitute a different cohort or an easier family if construction fails.

HF-A retains the intended role/persistence shift; HF-B retains causal-composition shifts. Both
must preserve the E3 two-object/two-container confusability and pass model-visible label-balance,
current-state leakage, matched-distractor, retention, endpoint and novelty audits. The current
`confusable_events_v3` generator applies its repaired construction only when `held_out_family`
is absent: its TEST path still falls back to legacy HF templates. Simply setting `test_episodes`
nonzero is NOT a valid next experiment. No old sealed test was opened to identify this code issue.

The proposed primary comparator is fixed chunks, the highest non-event E3 development mean.
This follows the original G1 strongest-development-baseline rule. E3's primary random-matched
contrast stays immutable; the new confirmation must always report random-matched as a mandatory
secondary contrast. Report all C0-C4 scores, tasks, lags, families and seeds, including failures.

The scoring implementation requires identical query/gold identities across conditions and seeds,
all 16 balanced cells, all 25 runs, and the 1,000-update endpoint. It calculates the existing
seed-then-paired-episode hierarchical interval, stratifying episodes by family and lag and keeping
all four task answers together. A seed-level t interval remains a sensitivity analysis. Review a
crossed/shared-episode bootstrap sensitivity before freeze because worlds repeat across seeds.

G1 thresholds are not relaxed: positive hierarchical interval, at least .05 absolute gain or
20% relative error reduction, positive effect in both families and in at least four seeds, and
all matched-resource controls. The new statistics API intentionally cannot grant G1: it does not
yet audit checkpoint/source/data provenance, FLOPs, latency or storage. C5 and visual transfer stay
closed. Profile/audit unequal retention horizons and segment lengths rather than assuming equal
record capacity guarantees equal historical coverage.

## B. Real-data companion: Sherlock dense development comparison

Use the same released 1,000 atomic movie-description rows and the existing 12-person development
recall split. The five test participants remain closed. This uses human annotations/recall, not
raw movie pixels, fMRI, Brain2Qwerty or TRIBE. It is a separate retrieval instrument, not a direct
transfer of the E3 neural world model into language.

Prepare 12 memory groupings: one fixed 20-row grouping, one human-event grouping, and ten
length-matched random groupings using the existing frozen random seeds. Each has 50 records.
Fine units remain a historical diagnostic, not an equal-capacity primary arm.

Encode each atomic description once and each eligible recall query once with one frozen encoder.
The encoder model/revision, dimension, tokenizer and long-text handling still need selection and
freezing. No model was selected, downloaded or run in this turn. Normalize atomic vectors, average
within each record, then normalize the record vector. All conditions reuse the same atomic cache;
no group-dependent re-encoding or silent truncation is allowed. Use the same cosine/dot ranker and
top five records. Scene/subject IDs and correctness labels never enter encoder text inputs.

The implemented record contract retains a float32 vector and two uint32 span bounds, with no source
text in the retrieved payload. Logical storage is `50 * (4 * dimension + 8)` bytes per condition.
This is NOT Python-process memory, encoder RAM, or total build cost. Report those separately.
Equal vector reads also do not mean equal source-text coverage; returned atomic-unit counts and
random/human length multisets must be reported.

Retain the v2 development primary: participant-macro maximum-IoU-target hit@5. Average the ten
random groupings within participant before any participant-level paired inference; random seeds
are not extra human subjects. Fixed chunks must always be reported. Ranking rows also expose
IoU-nDCG, maximum available IoU and returned atomic units.

Important remaining limitation: human scenes define both event grouping and evaluation targets.
Fixed-width embeddings remove a sparse-index/record-size confound, not that target-alignment bias.
A positive result would support recall-to-story retrieval on this instrument, not a claim that we
predict human remembering, explain fMRI, or beat all RAG. A negative primary cannot be replaced
with a nicer secondary metric. Full inference/provenance/reporting runner is not implemented yet.

## Implemented now

- `src/cmwm/evaluation/confirmation.py`: strict paired cohort arithmetic, hierarchical bootstrap,
  family/task/lag breakdowns and a structural guard rejecting legacy single-object held-out worlds.
- `src/cmwm/chen/sherlock_dense.py`: text-only encoder interface, float32 pooled-record construction,
  ranking and development scoring; sealed-subject and malformed-vector checks.
- `configs/next_stage/preparation.json`: draft counts, comparisons, disabled authorization flags
  and explicit unresolved work. Null provenance/model fields are deliberate blockers.
- `scripts/prepare_next_experiments.py`: prints the proposed inventory only. It has no launcher,
  execute flag, dataset/encoder reader or network call, and rejects enabled authorization flags.
- `tests/unit/test_next_stage_preparation.py`: tiny invented fixtures only; engineering checks,
  never experimental evidence. The 17 tests passed locally on 2026-09-30 using Python unittest.

Local-safe checks from the project root:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests/unit -p test_next_stage_preparation.py -v
PYTHONPATH=src python3 scripts/prepare_next_experiments.py
```

## Remaining before any submission request

1. Implement/audit new held-out families and the frozen checkpoint-evaluation adapter.
2. Review and freeze cohort/metric/bootstrap details, source and checkpoint identities, and budgets.
3. Pin the Sherlock encoder and long-text policy; implement provenance-checked development-only
   caching, full paired participant analysis, and an execution-guarded cluster runner.
4. Perform only authorized tiny local engineering tests until the user separately approves cluster
   engineering work. No draft inventory is itself permission to execute or access sealed data.

Slow/fast weights, consolidation, learned segmentation, vision and all other master-plan components
remain in the registry. They are not removed or silently added to this isolated memory comparison.
