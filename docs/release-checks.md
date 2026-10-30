# Release verification

Publication preparation on 30 October 2026 performs artifact checks, not new scientific experiments.

## Checks performed

- Original-to-export SHA-256 mapping for 495 selected files.
- All 25 E2 final runs: 512 unique aligned queries each, 64 in each task/lag cell, 12,800
  final prediction rows in total. Accuracy recomputed from saved predictions.
- E3 per-seed and taskwise means, final learning-curve values, paired contrasts and t intervals
  recomputed from the returned summary.
- Sherlock's primary null result and sealed-test flags checked.
- Source files parsed for syntax; current documentation links and SVG XML checked.
- Seventeen tiny deterministic preparation unit tests passed. These are software tests, not model results.
- Seven figures rendered and visually inspected; label bounds checked in a local browser.
- Publication copy scanned for credential patterns, private absolute paths and private account references.

## What was not rerun

No training, cluster submissions, dataset-scale generation, model inference, raw neural-data analysis,
sealed-test evaluation or external model API calls were performed. The full tensor-dependent test
suite was not rerun in the packaging environment. Historical pass counts remain historical.

E3 checkpoint tensors and original prediction files remain outside this release. Their contents
were not re-evaluated during publication. Statistical arithmetic verification is not a new replication.

Reproduce the release checks with `python scripts/verify_release.py`. After intentional changes,
review them before using `python scripts/seal_release.py` to create a new release-file hash index.
