> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# FilmFest Reproduction Report V1

Date: 2026-08-11  
Scope: Phase 1A behavioral/event normalization and released Figure 3 result reproduction  
Paper: Lee & Chen (2022), <https://doi.org/10.1038/s41467-022-31965-2>

## Bottom line

The released Figure 3 event-level and participant-level statistics reproduce within the predeclared absolute tolerance of 0.015. All six comparisons pass. The normalized dataset contains 202 unique events from 10 films, with descriptions from all three released annotators and exact movie-specific recall counts reconstructed from the released scene-matched recall files.

This is a **downstream reproduction from released figure data**, not a complete upstream reconstruction of the semantic and causal graphs. Gate A is therefore **partially passed**:

- behavioral/statistical reproduction: **PASS**;
- event and recall normalization: **PASS**;
- exact semantic graph reconstruction: **BLOCKED pending a pinned executable 2018 USE environment/model artifact**;
- exact causal graph reconstruction: **BLOCKED because coder-level event-pair judgments are absent from the located public release**.

Proceeding as if the two published centrality columns were newly reconstructed would be misleading.

## Reproduced results

| Result | Published | Reproduced | Absolute difference | Status |
|---|---:|---:|---:|---|
| Semantic centrality–recall Pearson r | 0.20 | 0.202424 | 0.002424 | PASS |
| Causal centrality–recall Pearson r | 0.29 | 0.294099 | 0.004099 | PASS |
| Semantic high–low paired t | 6.12 | 6.119081 | 0.000919 | PASS |
| Causal high–low paired t | 8.23 | 8.229111 | 0.000889 | PASS |
| Semantic high–low Cohen's dz | 1.58 | 1.579940 | 0.000060 | PASS |
| Causal high–low Cohen's dz | 2.12 | 2.124747 | 0.004747 | PASS |

Additional recomputed values:

- semantic correlation: n=202, two-sided p=0.003864, Fisher 95% CI [0.0662, 0.3312];
- causal correlation: n=202, two-sided p=0.0000215, Fisher 95% CI [0.1627, 0.4153];
- semantic high–low: n=15, mean difference=0.09297, two-sided p=0.0000265;
- causal high–low: n=15, mean difference=0.13379, two-sided p=0.000000984.

Machine-readable results are in `behavioral_statistics_v1.json` (historical source location: `./behavioral_statistics_v1.json`), `reproduction_evaluation_v1.json` (historical source location: `./reproduction_evaluation_v1.json`), and `published_comparison_v1.csv` (historical source location: `./published_comparison_v1.csv`).

## Plots

- `semantic_centrality_vs_recall.svg` (historical source location: `./plots_v1/semantic_centrality_vs_recall.svg`)
- `causal_centrality_vs_recall.svg` (historical source location: `./plots_v1/causal_centrality_vs_recall.svg`)
- `semantic_high_low_recall.svg` (historical source location: `./plots_v1/semantic_high_low_recall.svg`)
- `causal_high_low_recall.svg` (historical source location: `./plots_v1/causal_high_low_recall.svg`)

The scatterplots show the 202 released event values colored by film, with an ordinary least-squares line. The paired plots show all 15 released participant rows and the two group means. They are descriptive reproductions; inferential values come from the scripts, not visual fitting.

## Normalized event and recall data

The normalized table is `normalized_events.csv` (historical source location: `../../data/filmfest/normalized_events.csv`), with schema `normalized_event.schema.json` (historical source location: `../../data/filmfest/normalized_event.schema.json`) and provenance `normalized_events.metadata.json` (historical source location: `../../data/filmfest/normalized_events.metadata.json`).

Validation checks:

- 202 rows and 202 unique event IDs;
- 10 stories with published event counts: 23, 14, 25, 12, 25, 27, 16, 10, 15, and 35;
- all 202 events have non-empty descriptions from JL, KM, and RC;
- original run-local segment number and global scene-matching ID are both preserved;
- published event recall probability is preserved together with reconstructed integer numerator and denominator;
- normalized CSV SHA-256: `7b223fadd53b8937fc930504e3f40e91929443eb976d7a8a550581ab8809f7a0`.

Recall probabilities are conditional on a movie-specific event-analysis cohort, not uniformly on all 15 participants. The exact denominators recovered from the scene-matching files are 15, 14, 15, 12, 14, 15, 13, 10, 11, and 14. The unique largest-cohort matching check identifies two discrepancies that must remain visible:

- Keith Reynolds: subject 07 has specific-scene coding in the released workbook but is excluded from the 14-person cohort implied by every published event probability;
- The Black Hole: subject 12 has specific-scene coding but is excluded from the 10-person cohort implied by every published event probability.

The release does not state the reason for these two exclusions in the inspected files. The preprocessing script records both subject lists rather than inventing an explanation.

## Source provenance

- official FilmFest repository clone HEAD: `0ffad622cba677484890d47082d1bc351b38bfe1`;
- immutable FilmFest `v1.0.0` tag: `ca5b3f5e8afce70f3f38724b2e9fee74b76803ae`;
- official Source Data workbook SHA-256: `e1350ba6c98aeafbab0fda427482b000b0caa95eba75daed9fd8c2ef2a3b021e`;
- JL annotation SHA-256: `a35ce12aa5c6690da03f7105eefd871336ba729adce29218713615179aeab343`;
- KM annotation SHA-256: `988d601e67069fc4aa80db04983afcfb1c514cbcfef4b4be2678a5539a2dd62d`;
- RC annotation SHA-256: `97dcdd3395b4a530c335c83bbda6cc60174dc0cb534855a674c393befc0436d3`.

The three annotation workbook blobs are identical between the `v1.0.0` tag and current `main`. The tag does not contain `recall_scenematched/`; current `main` adds the 20 recall/coding workbooks used for exact recall-count recovery. Both refs are retained so release provenance is not collapsed into a single ambiguous version.

The GitHub-only workbooks do not have an explicit standard license in the audited repository. The paper and its Source Data are CC BY 4.0. Redistribution rights for GitHub-only data should be clarified before public packaging.

## What was and was not reconstructed

### Reconstructed

1. Coarse event order after removing each six-second movie-title segment.
2. Global scene-matching IDs across both scan runs.
3. Event descriptions for all three annotators.
4. Movie-specific event-recall cohorts, integer counts, and probabilities.
5. Correlations of released semantic/causal centrality with recall.
6. Participant-level high-versus-low centrality comparisons.

### Not reconstructed upstream

1. **Semantic edges.** The paper averages 512-dimensional Universal Sentence Encoder representations across three annotators and uses within-film cosine similarity. The graph builder accepts pinned event embeddings and implements the unthresholded cosine graph, but no substitute embedding model was used. A modern replacement would be a new experiment, not an exact reproduction.
2. **Causal edges.** The paper describes coder-level pair judgments and edge weights as endorsement proportions. The official GitHub repository, Zenodo bundle, and Source Data archive inspected here expose causal centralities but not those raw judgments. The causal graph builder therefore requires the missing input and fails loudly when it is absent.
3. **Published centrality derivation.** The weighted-degree transformation is implemented, including within-film normalization and MATLAB-default sample-SD z-scoring, but cannot be applied to exact source graphs until the preceding inputs exist.

## Reproducible commands

Run from `outputs/cewm` using the workspace Python recorded in the JSON artifacts:

```bash
python reproduction/filmfest/prepare_data.py \
  --annotations-dir data/filmfest/source_repo/annotations \
  --recall-dir data/filmfest/source_repo/recall_scenematched \
  --source-data data/filmfest/source_data/extracted/SourceData.xlsx \
  --source-commit 0ffad622cba677484890d47082d1bc351b38bfe1 \
  --output data/filmfest/normalized_events.csv \
  --metadata-output data/filmfest/normalized_events.metadata.json

python reproduction/filmfest/reproduce_recall_results.py \
  --source-data data/filmfest/source_data/extracted/SourceData.xlsx \
  --output results/filmfest_reproduction/behavioral_statistics_v1.json \
  --plot-dir results/filmfest_reproduction/plots_v1

python reproduction/filmfest/evaluate_reproduction.py \
  --results results/filmfest_reproduction/behavioral_statistics_v1.json \
  --output results/filmfest_reproduction/reproduction_evaluation_v1.json \
  --table-output results/filmfest_reproduction/published_comparison_v1.csv
```

Scripts refuse to overwrite existing derived artifacts. A clean rerun should use a fresh output directory or versioned filenames.

## Decision and next exact task

Do not declare Gate A fully cleared and do not start ERCM. First request the raw FilmFest causal event-pair judgments and the authors' exact semantic-embedding environment/model artifact or cached embeddings. If those inputs cannot be obtained, freeze the limitation and decide explicitly whether released centralities are sufficient for the oracle-graph opportunity analysis. That decision would change the evidentiary strength of Phase 1A and must be approved before Task 11 or the Phase 2 benchmark draft proceeds.
