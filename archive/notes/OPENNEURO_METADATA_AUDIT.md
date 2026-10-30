> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# OpenNeuro Metadata Audit for the Chen Lab Scope

Audit date: 2026-08-16

Cluster verification: CPU job `1753886` completed with three fixture tests
passing. Returned JSON SHA-256:
`2846253e054a196bfa92be0c1d4bda1d541e39d185a5be37f7e1d129906505c6`.

Method: read-only GraphQL queries to the official OpenNeuro endpoint documented
at <https://docs.openneuro.org/api.html>. The audit queried immutable snapshot
metadata and recursive file-tree declarations. It did not download imaging or
stimulus files.

| Dataset | Snapshot | Dataset name | Authors | Top-level subjects | Recursive files | Declared bytes | Snapshot license field |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| FilmFest | `ds004042:1.0.0` | `FilmFestival` | Hongmi Lee; Janice Chen; Uri Hasson | 20 | 180 | 41,721,639,565 | `CC0` |
| Sherlock | `ds001132:1.0.0` | `Sherlock` | Chen, Leong, Honey, Yong, Norman, Hasson | 16 | 121 | 6,606,797,171 | empty string |
| The Twilight Zone | `ds001145:1.0.0` | `Twilight Zone Movie Watching Dataset` | Chen, Honey, Simony, Arcaro, Norman, Hasson | 24 | 81 | 8,788,116,436 | empty string |
| Merlin + Sherlock Listening | `ds001110:00003` | `Sherlock_Merlin` | Zadbood, Chen, Leong, Norman, Hasson | 37 | 287 | 9,506,919,942 | null |

The byte totals are sums of the sizes declared by OpenNeuro's recursive snapshot
tree. They are planning estimates, not checksums or evidence that any local copy
is complete. The listed subject counts are directory counts in these exact
snapshots and should not be substituted for paper-specific analytic sample
sizes after exclusions.

## File-shape findings

- FilmFest declares 145 annexed `.nii.gz` files, 28 TSV files, and four JSON
  files. Its top level contains separate movie and recall task metadata and 20
  participant directories.
- Sherlock declares 64 `.nii.gz` files, 47 TSV files, four JSON files, one
  uncompressed NIfTI file, derivatives, and 16 participant directories. Its top
  level distinguishes `sherlockPart1`, `sherlockPart2`, and `freerecall` tasks.
- The Twilight Zone declares 48 `.nii.gz` files, 24 TSV files, two JSON files,
  two M4V files, and 24 participant directories. Its top-level task is
  `watchmovie`.
- Merlin + Sherlock Listening declares 114 `.nii.gz` files, 114 TSV files, 43
  JSON files, eight WAV files, one MP4, one M4V, and 37 participant directories.
  Its task metadata explicitly distinguishes Merlin audio, movie, and free
  recall and Sherlock audio, movie, and free recall.

## Acquisition decision

Do not bulk-download these approximately 66.6 GB of declared files yet.
FilmFest and Sherlock already have the lightweight annotations/behavior needed
for the first memory work. Merlin + Sherlock Listening first needs a file-level
stimulus/license audit and a separately frozen cross-modal hypothesis. When
neural data become necessary, the exact subject/task files will be selected and
checksummed on CLSP.

## Narratives cross-reference

OpenNeuro `ds002345:1.1.4` is a later 345-participant aggregate of 28 auditory
stories. Its immutable participant table contains 82 unique people with a
`pieman` task and 23 with a `tunnel` task, and its stimulus inventory contains
`pieman_audio.wav` and `tunnel_audio.wav`. Its README restricts the audio clips
to non-profit, non-commercial scholarship/research under fair-use or
fair-dealing principles. This aggregate is useful for provenance and raw-BIDS
access, but it does not erase the identity/version of the official Princeton
releases.

## Unresolved Princeton metadata

The Princeton-hosted records are verified as links on the official Chen Lab
dataset index, but Princeton DataSpace currently places an interactive
browser-verification layer in front of both landing pages and its OAI endpoint.
OpenNeuro independently supplies snapshot metadata for Sherlock and Twilight
Zone, and Narratives supplies a cross-reference for Pieman and Tunnel. The
Princeton-specific derivative inventories and the Green Eyes metadata remain
unverified; no participant, byte, file, or license values were guessed.
