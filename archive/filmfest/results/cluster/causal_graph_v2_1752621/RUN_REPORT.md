> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# FilmFest Directed Causal Graph V2 — CLSP Run Report

## Outcome

The run produced a complete **model-predicted directed graph** for the ten FilmFest narratives. It is suitable for the `predicted_graph` experimental condition and for preparing graph-based retrieval questions. It is **not** an independent human causal oracle.

## Frozen construction rule

- Candidate universe: every temporally forward event pair within each story.
- Candidate pairs: 2,226 across 202 events and 10 stories.
- Rating scale: 0–3 causal strength.
- Retained edge: score at least 2.
- Direction: earlier event to later event.
- Weight: score divided by 3.
- No semantic-similarity, recall, centrality, or outcome-based prefilter was used.
- The forward-only rule makes the retained graph a DAG by construction.

## CLSP execution

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Pinned revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- Accelerator: NVIDIA A100-SXM4-80GB
- Precision: BF16
- Main Slurm array: job `1752621`, four shards, with at most three GPUs running concurrently because of the available QOS.
- Main shard runtimes: 4:05, 4:16, 4:58, and 4:08.
- Targeted repair jobs: `1752625`–`1752628`.
- Final CPU aggregation: job `1752630`, four CPUs, one second.

## Integrity and coverage

- Rated candidate pairs: 2,226 / 2,226.
- Raw attempt rows preserved: 2,230.
- Invalid attempts preserved: 4.
- All three initially unresolved pairs received valid targeted repairs.
- Final retained edges: 575 / 2,226 (25.83%).
- Edge CSV SHA-256: `9e8418d7f0f0f5b273c13976af33d95dd305ddc86f4714309c2b74905919f8ce`.
- Metadata SHA-256: `96b6b511001a2e2f5095f2594921c87cc7185f62238597acf5aab3b4225fc493`.
- All downloaded aggregate checksums passed.

## Score distribution

| Score | Pair count |
|---:|---:|
| 0 | 1,303 |
| 1 | 348 |
| 2 | 554 |
| 3 | 21 |

## Retained relation labels

| Relation | Edge count |
|---|---:|
| motivational | 362 |
| enables | 150 |
| informational | 38 |
| goal_action | 18 |
| physical | 6 |
| other | 1 |

## Per-story density

| Story | Candidates | Edges | Density |
|---|---:|---:|---:|
| bus_stop | 595 | 159 | 0.267 |
| catch_me_if_you_can | 253 | 74 | 0.292 |
| keith_reynolds | 300 | 92 | 0.307 |
| post_it_love | 105 | 30 | 0.286 |
| the_black_hole | 45 | 18 | 0.400 |
| the_boyfriend | 300 | 112 | 0.373 |
| the_prisoner | 120 | 15 | 0.125 |
| the_record | 91 | 40 | 0.440 |
| the_rock | 351 | 33 | 0.094 |
| the_shoe | 66 | 2 | 0.030 |

The wide density range and the concentration of `motivational` labels are reasons to audit calibration rather than treating the graph as human ground truth.

## Retrieval-path yield

- Anchors with at least one causal ancestor: 143.
- Eligible local-path anchors: 143.
- Eligible medium-path anchors: 126.
- Eligible long-path anchors: 74.

This is enough structural yield to build balanced graph-distance retrieval tests, subject to answer-evidence and leakage checks.

## Interpretation boundary

This graph measures what the pinned Qwen model inferred from the released event descriptions under the frozen prompt. It does not reproduce the unreleased FilmFest coder matrix, and it must not be labeled as the paper's graph or used as an `oracle_graph` condition. An independent human graph can become the V2 oracle only after at least two blinded annotators rate the same complete candidate universe, disagreements are adjudicated, and inter-rater reliability is reported.

## Next decision gate

1. Keep this artifact frozen as `model_predicted_graph_v2`.
2. Collect two independent human rating passes with the blank workbook.
3. Report agreement and adjudicate disagreements without consulting model labels.
4. Compare predicted and human graphs only after the human graph is frozen.
5. Build retrieval questions from both graphs, but label their experimental conditions separately.
