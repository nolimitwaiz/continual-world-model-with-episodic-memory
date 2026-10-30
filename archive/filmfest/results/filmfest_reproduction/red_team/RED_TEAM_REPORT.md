> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# FilmFest red-team validation report

Date: 2026-08-11  
Scope: Chen Lab FilmFest data only; Sherlock and ERCM experiments were not started

## Bottom line

The released FilmFest behavioral data are small but usable for the intended Phase 1A sanity
check. The participant/event mapping, cohort reconstruction, Figure 3 statistics, and centrality
effects survive independent checks. The project is not yet entitled to claim an exact upstream
graph reproduction: raw causal judgments and exact semantic embeddings remain unavailable.

The appropriate decision is therefore:

- behavioral/statistical Gate A: **strong pass**;
- exact semantic graph reconstruction: **blocked on exact USE vectors/environment**;
- exact causal graph reconstruction: **blocked on raw coder judgments**;
- H2/ERCM or benchmark superiority: **not tested**.

## What data were actually used

### Main fMRI/free-recall cohort

- 15 released participants;
- 10 films;
- 202 annotated events;
- 3,030 possible participant-event rows;
- 2,778 rows in the recovered published event-analysis cohorts;
- movie-specific denominators from 10 to 15, not a uniform denominator of 15.

The raw participant workbooks contain scene-matched recall codes. The Nature Source Data workbook
contains released centralities and aggregate/event-level results. No raw fMRI volumes were
downloaded or analyzed in this validation.

### Independent online cohort

- 492 raw respondents;
- 393 included respondents;
- 85 excluded for short recall and 14 for prior exposure;
- 10 different films and 252 events.

The raw online workbook contains demographics and free-recall text but not the event-level recall
labels used in the paper. Participant-level online mixed-effects results therefore cannot be
reconstructed from the public files. The released Source Data still permits an aggregate event
audit.

## Mapping and missingness checks

The new `recall_matrix.csv` preserves three distinct states: a coded recalled event, a coded
non-recalled event, and a missing participant/movie combination. This avoids the serious error of
treating every event in an unrecalled movie as a zero.

Four manual source-to-matrix spot checks covered ordinary and exceptional cases:

- subject 01, movie 1;
- subject 20, movie 10;
- subject 07, movie 5;
- subject 12, movie 8.

Scene IDs and recalled flags matched the source workbooks. The last two cases are specifically
scene-coded in the release but are excluded by the unique largest cohorts that reproduce all
published event probabilities. This accounts for 35 participant-event rows: 25 for Keith Reynolds
and 10 for The Black Hole. The public release does not explain those exclusions, so the matrix
labels them as inferred published-cohort exclusions rather than inventing a reason.

## Cohort sensitivity

The reconstructed published-cohort probabilities match the released values to four-decimal
rounding, with maximum absolute residual `0.0000462`. Including the two scene-coded exclusions
changes 35 of 202 event probabilities; the maximum event-level change is `0.0909`.

| Recall definition | Semantic r | Causal r |
|---|---:|---:|
| Recovered published cohort | 0.2024 | 0.2941 |
| All specifically scene-coded participants | 0.1953 | 0.2877 |
| All 15 participants, with non-coded movies counted as zero | 0.1916 | 0.2775 |

The effect is modestly attenuated under broader denominator rules but remains positive. The final
row is a stress test, not the preferred analysis, because it turns missing/unrecalled movies into
event-level zeros.

## Participant high-versus-low reconstruction

Using `round(0.4 × number of events)` within each film and averaging recall proportions over each
participant's included films:

- all 15 semantic high/low rows match the released Figure 3b values within four-decimal rounding;
- 14 of 15 causal rows match Figure 3d;
- the remaining causal high value differs by `0.0143` for released subject index 12.

That isolated difference is traceable to ties at the top-40% cutoff in the rounded published causal
centralities. The exact graph weights or the authors' selected event list are needed to resolve the
tie. The discrepancy is retained rather than choosing a tie ordering solely to reproduce the plot.

## Independent statistical checks

SciPy independently reproduces the released-figure statistics:

| Check | Independent value |
|---|---:|
| Semantic centrality–recall Pearson r | 0.202424 |
| Causal centrality–recall Pearson r | 0.294099 |
| Semantic paired high–low t | 6.119081 |
| Causal paired high–low t | 8.229111 |

A participant-event logistic sensitivity model with participant and movie fixed effects produced:

| Effect | Beta | LRT chi-square | Paper rounded beta | Paper rounded chi-square |
|---|---:|---:|---:|---:|
| Semantic centrality | 0.1760 | 12.3820 | 0.17 | 12.24 |
| Causal centrality | 0.3844 | 55.5759 | 0.38 | 55.04 |

This is not the paper's lme4 random-intercept model, so near-agreement is a robustness check rather
than an exact model reproduction. A separate statsmodels variational-Bayes random-intercept check
gave posterior means of `0.1747` for semantic and `0.3826` for causal centrality, again consistent
in direction and magnitude.

For the independent online films, the released aggregate data give event correlations of `0.2479`
for semantic and `0.3378` for causal centrality. An aggregate binomial model with movie fixed
effects gives betas `0.1664` and `0.4204`, near the paper's rounded participant-level values of
`0.17` and `0.44`. It is not an exact replacement for the unavailable participant-event model.

## Compute and reproducibility

The independent-model run was CLSP Slurm job 1752543 on CPU node `octopod`: four CPUs, 8 GB RAM,
no GPU, completed in 22 seconds. A GPU would not help these small spreadsheet and regression jobs.
The environment pinned Python 3.12.13, SciPy 1.16.1, and statsmodels 0.14.5. Downloaded artifacts
passed SHA-256 verification. The full local suite now contains 14 passing tests.

## What these findings do and do not establish

They establish that the released event mapping is recoverable, the unusual movie denominators are
not an accidental parsing artifact, the centrality/recall association is robust to reasonable
cohort perturbations, and independently fitted models closely track the reported behavioral
effects.

They do not establish that causal structure is mechanistically responsible for recall, that the
semantic or causal graphs have been reconstructed from raw inputs, that ERCM improves retrieval,
or that the effects generalize to Sherlock. Published centrality values remain oracle-derived
inputs until the missing upstream graph materials are obtained.

## Recommended next action

Send the prepared author-data request for raw causal judgments, cached USE embeddings or the exact
USE artifact/environment, and the event lists used for tied high/low cutoffs. While waiting, the
only scientifically safe continuation is to draft the benchmark specification with released
centralities explicitly labeled as oracle inputs. Building the benchmark or starting ERCM should
wait for Waiz's explicit fallback approval because that changes the strength of the Phase 1A
reproduction claim.
