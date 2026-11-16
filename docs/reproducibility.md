# Reproduction and verification

## Three levels of reproducibility

1. **Release checks:** hashes, result arithmetic, links and static source validation. No training.
2. **Software tests:** tiny deterministic fixtures plus the full unit suite when dependencies exist.
3. **Scientific reruns:** require a configured compute environment, matching manifests and separately
   frozen protocols. This release is not an automatic one-command reproduction of all original runs.

## Minimal artifact verification

From the repository root, Python 3.11+:

```bash
python scripts/verify_release.py
python scripts/verify_research_summary.py
python scripts/build_figures.py
PYTHONPATH=src python -m unittest discover -s tests/unit -p test_next_stage_preparation.py
python -m unittest discover -s tests/unit -p test_saved_endpoint_sensitivity.py
```

No external packages are required by the release checkers or SVG figure builder. The 17 preparation
tests and 9 scoring-audit tests use constructed fixtures, not model training. The summary checker
also recalculates the saved scoring audit, the Sherlock random-group mean, and all six FilmFest
reproduction checks. It does not recover missing E3 predictions or provide new-world validation.

## Optional Figure Export

The SVG figures are the editable source images and are generated directly from saved JSON.
For PNG copies, install Node.js and the `sharp` package in your own environment, then run:

```bash
node scripts/render_figures.cjs
node scripts/render_figures.cjs --names overview --output docs/figures
```

The first command writes inspection images under the ignored `.preview/figures` directory.
The second regenerates the overview diagram's PNG copy. Neither command runs a model.

## Full software environment

On an appropriate research machine:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest tests/unit
cmwm validate-config --config configs/e3/c4_oracle.yaml
```

The dependency ranges in `pyproject.toml` are software compatibility ranges, not an exact historical
environment lock. E2/E3 artifacts record PyTorch 2.5.1+cu121 and A100-SXM4-80GB hardware. The full
tensor-dependent suite was not rerun during packaging when those dependencies were unavailable.
Previously recorded test counts are historical results, not current CI guarantees.

## Scientific execution

The original workflow requires Slurm, CUDA, explicit data/run directories and an attestation matching
the actual source/configuration. The [submission utility](../scripts/submit_slurm.py) supports dry-run
inspection and separately requires a confirmation flag and authorization environment variable for
real submission. No launch authorization is supplied in this release and no CI training is configured.
A hardware-specific environment script and old retry manifests were excluded rather than published
with active authorization or personal paths.

Configuration inspection is safe; a dry run can be inspected after dependencies are installed:

```bash
python scripts/submit_slurm.py --config configs/e3/c4_oracle.yaml --dry-run
```

To plan a scientific rerun, provision compute, reconstruct the exact declared data/endpoint,
verify provenance and leakage controls, and only then obtain explicit execution authorization.
Do not disable safeguards simply to make a configuration launch. The current code includes later
preparation changes and must not be assumed byte-identical to the original training source.

## Confirmation blockers

- The E3 TEST branch uses legacy held-out templates, not the repaired confusable world construction.
  It must not be advertised as a valid independent E3 confirmation set.
- Held-out-family generation, retention/resource audits and independent protocol freezing remain.
- Five seeds on one reused validation cohort do not supply new-world uncertainty estimates.
- Sherlock dense retrieval still lacks a complete frozen encoder/cache/runner and paired participant
  analysis. The five-person test cohort remains sealed.
- E3's original checkpoints and per-checkpoint prediction artifacts are not included locally.

## Data acquisition

Obtain third-party data from the original providers under their terms; see [data documentation](data.md).
No raw brain scans, participant recall text, films, credentials or private correspondence are redistributed.
