# Saved Results

For interpretation, start with [Results and plots](../docs/results.md). These files are
returned measurements and later analyses, not a claim that every recorded job is an
independent experiment.

- [E2 final endpoints](cluster/1779882_e2_final/): five conditions and five seeds, including
  saved predictions used to recompute the original score.
- [E3 returned summary](cluster/1806501_e3_final/summary.json): five-seed results, task-level
  scores, and learning curves. Original checkpoints and per-question outputs are not included.
- [Sherlock corrected development result](cluster/1764075/result.json): aggregate text-retrieval
  comparisons, including the negative main result and unused-test flags.
- [Later scoring audit](analysis/saved_endpoint_sensitivity_2026-11-11/REPORT.md): separates
  ordinary answers from source-record selection without replacing the original main measure.
- [FilmFest reproduction](../archive/filmfest/results/filmfest_reproduction/reproduction_evaluation.json):
  checks of six released statistics, not new model-training results.
- [Complete artifact catalog](../docs/artifact-catalog.md): engineering runs, earlier development,
  and graph diagnostics with their limitations.

Use `python scripts/verify_release.py` and `python scripts/verify_research_summary.py`
from the repository root to check saved artifacts. Neither command trains a model.
