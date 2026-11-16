# Release verification

The original publication copy was prepared on 30 October 2026. The 16 November update
improves the research-facing documentation and adds a saved-result scoring audit, a
short proposal, and clearer diagrams. These are artifact checks, not new experiments.

## Checks performed

- Original-to-export SHA-256 mapping for 495 selected files.
- All 25 E2 final runs: 512 unique aligned queries each, 64 in each task/lag cell, 12,800
  final prediction rows in total. Accuracy recomputed from saved predictions.
- E3 per-seed and taskwise means, final learning-curve values, paired contrasts and t intervals
  recomputed from the returned summary.
- Sherlock's primary null result and sealed-test flags checked.
- Source files parsed for syntax; current documentation links and SVG XML checked.
- Seventeen deterministic preparation tests and nine scoring-audit tests passed in the
  16 November packaging environment. These are software tests, not model results.
- The 11 November scoring audit was recomputed from included E2 predictions and E3 summaries.
- Sherlock's ten random-grouping scores were averaged again; all six FilmFest reproduction
  comparisons were checked against their declared tolerance. Headline documentation values
  were checked against those artifacts.
- Nine diagrams/plots and the two-page proposal were generated. Figures and PDF pages
  were rendered and visually inspected before release.
- Publication copy scanned for credential patterns, private absolute paths and private account references.

## What was not rerun

No training, cluster submissions, dataset-scale generation, model inference, raw neural-data analysis,
sealed-test evaluation or external model API calls were performed. The full tensor-dependent test
suite was not rerun in the packaging environment. Historical pass counts remain historical.

E3 checkpoint tensors and original prediction files remain outside this release. Their contents
were not re-evaluated during publication. Statistical arithmetic verification is not a new replication.

Reproduce the checks with `python scripts/verify_release.py` and
`python scripts/verify_research_summary.py`. After intentional changes, review and stage
only the files intended for publication, then use
`python scripts/seal_release.py --release-date YYYY-MM-DD` to regenerate the hash index.
The index covers tracked/staged files only, not private notes or untracked drafts.
