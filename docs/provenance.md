# Artifact provenance and release coverage

## What is included

This repository is a publication copy of the research workspace, not a rewrite of its originals.
It includes the current source/configuration/test tree, returned compact numeric JSON artifacts,
E2 final synthetic predictions, E3 per-seed and taskwise summary, Sherlock aggregate results,
and dated research reports. Historical FilmFest reproduction and exploratory diagnostics are
kept separately. The [catalog](artifact-catalog.md) indexes these records.

## What is not included

- Raw human recall text, participant workbooks, films and brain scans.
- Model checkpoints or the complete original E3 per-checkpoint prediction files.
- Private correspondence, account credentials and machine-specific launch environments.
- Verbose cluster logs, hardware-node inventories, caches, private workflow notes and redundant copies.
- The complete legacy FilmFest software/data environment. Historical reports alone do not make that
  earlier pipeline fully reproducible from this repository.

All locally returned compact results from the current project are retained subject to these exclusions.
"All results" does not imply that cluster-only artifacts or restricted upstream datasets are present.

## Transformations

[export_manifest.json](../provenance/export_manifest.json) records original and released SHA-256
digests for copied files. Private absolute paths and machine names are replaced with placeholders.
JSON is reformatted without altering numerical values. Historical notes receive an explicit status
header; links to omitted local data are shown as historical locations rather than broken download links.
Incidental workflow branding is omitted from the publication copy. Scientific caveats and unfavorable
results are preserved.

Consequently, an original `config_sha256` or `source_sha256` embedded inside an artifact still identifies
the historical experiment, not necessarily the sanitized file or current source snapshot. Do not
silently substitute new hashes into historical experiment identities. The release manifest and
[file hashes](../provenance/release_files.json) describe this publication snapshot separately.

The 16 November documentation update adds the 11 November saved-endpoint sensitivity
analysis, its source-checking script and tests, and a short research proposal. No original
measurement is overwritten. The release-file inventory now covers explicitly tracked
publication files only; untracked local plans, private notes, and caches are excluded.

## Statistical coverage

E2 endpoint accuracy can be checked against included final synthetic predictions. E3 arithmetic
can be checked from per-seed scores and taskwise summaries, but original E3 model outputs cannot be
re-derived without the omitted predictions/checkpoints and original environment. Sherlock summaries
can be compared, but participant-level retrieval cannot be rerun without obtaining original data.

## Rights and attribution

Original providers retain rights to their datasets, papers and third-party software. No new
open-source license is selected by this release. Repository access is not a grant of rights to
redistribute excluded data. Institutional or laboratory endorsement is not implied.
