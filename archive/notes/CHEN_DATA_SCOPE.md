> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# Chen Lab Data Scope

Decision date: 2026-08-16

## Binding scope

The active external-data program is limited to the seven releases listed in the
**Datasets** section of the Janice Chen Lab website:

1. **Film Festival (FilmFest)** — OpenNeuro `ds004042`, snapshot `1.0.0`, plus
   the lab's lightweight annotations and recall products.
2. **Sherlock** — OpenNeuro `ds001132`, snapshot `1.0.0`, plus the Princeton
   preprocessed release, tutorial annotations, and recall transcripts.
3. **The Twilight Zone** — Princeton DataSpace handle
   `88435/dsp01bv73c289q`; the same study also has OpenNeuro snapshot
   `ds001145:1.0.0`.
4. **Pieman** — Princeton DataSpace handle `88435/dsp015d86p269k`.
5. **Tunnel Under the World** — Princeton DataSpace handle
   `88435/dsp011n79h6771`.
6. **Green Eyes** — Princeton DataSpace handle `88435/dsp0141687k93v`.
7. **Merlin + Sherlock Listening** — OpenNeuro `ds001110`, snapshot `00003`.

The official index is <https://jchenlab.johnshopkins.edu/>. The machine-readable
inventory is `CHEN_DATASET_MANIFEST.csv` (historical source location: `CHEN_DATASET_MANIFEST.csv`).
The three OpenNeuro snapshots have also received a metadata-only file audit in
`OPENNEURO_METADATA_AUDIT.md` (historical source location: `OPENNEURO_METADATA_AUDIT.md`); no imaging data
were downloaded.

Chen-associated resources such as Narratives, Agency, or later lab papers are
useful references, but they are not silently merged into this core seven. They
require a separate decision if a specific hypothesis needs them.

## What this does and does not mean

The seven datasets are a **benchmark family**, not one homogeneous training
corpus. They contain different stimuli, tasks, participants, timing systems,
annotations, and preprocessing histories. Pooling raw rows or fMRI volumes
would create leakage and destroy interpretability.

We will therefore share only a versioned interface:

```text
source record
  -> immutable dataset/story/participant identifiers
  -> stimulus and time coordinate
  -> event interval and description, when available
  -> recall/reinstatement observation, when available
  -> relation provenance, when available
  -> dataset-specific neural pointer, when needed
```

Every derived record must retain its source dataset, story, participant,
version, checksum, and transformation provenance. Story- and participant-level
splits remain separate. Neural data are never treated as independent random
samples at the TR/voxel level.

## Research order

### 1. FilmFest: primary real-data benchmark

Use the released event descriptions and recall alignments to test the immediate
question: as semantic overlap decreases and causal distance increases, does
event/causal reinstatement beat similarity retrieval under matched evidence and
compute budgets?

The released statistics have already been reproduced and independently checked
on CLSP. The confirmatory retrieval benchmark remains blocked on verified human
directed causal edges and its frozen question manifest. The existing
model-predicted graph is exploratory and cannot substitute for that oracle.

### 2. Sherlock: external event-memory validation

Use its long narrative, human scene/event labels, and view-to-recall alignment
to test event segmentation and reinstatement without the FilmFest graph. This
is the first cross-dataset check that a FilmFest result is not a property of one
annotation scheme.

### 3. Twilight Zone: long-timescale access

Use it only after its files, timing, participants, and reuse constraints are
inventoried. Its role is a minutes-versus-hours memory-access test, not a source
of extra training rows for FilmFest.

### 4. Pieman, Tunnel Under the World, Green Eyes, and Merlin + Sherlock Listening

Use these as held-out narratives for shared-response, semantic-context, and
cross-story generalization studies. Their exact role is frozen only after a
metadata/file audit shows which event, transcript, behavioral, and neural
products are actually present.

Pieman and Tunnel also occur in the later OpenNeuro Narratives aggregate
(`ds002345:1.1.4`). That aggregate is an access/provenance cross-reference, not
permission to redefine the two official Chen Lab releases. Its participant
table contains 82 people with a `pieman` task and 23 with a `tunnel` task.

## Acquisition policy

1. Audit landing-page metadata, exact snapshot/version, file tree, sizes,
   checksums, license, stimulus restrictions, and participant/task structure.
2. Acquire annotations, transcripts, timing tables, and behavioral products
   before large neuroimaging assets.
3. Download raw or preprocessed fMRI only when a preregistered neural hypothesis
   names the exact files and analysis unit.
4. Keep copyrighted film/audio stimuli outside redistributed derivatives even
   when a BIDS snapshot is public.
5. Run executable QC and all experimental computation on CLSP; local work is
   limited to documentation, static validation, and inspection of returned
   artifacts.

## Immediate gates

1. G0 is complete: CLSP job `1753878` passed on 12,000 episodes and 97,548
   queries. This validates the controlled memory testbed; it is not model
   training or evidence about human data.
2. Complete the seven-dataset source/file inventory; Green Eyes and the
   Princeton-specific derivative inventories remain unresolved.
3. Finish the FilmFest human-edge/question-manifest requirement or explicitly
   register a separate predicted-graph exploratory experiment.
4. Run FilmFest baselines, then ERCM V0, under the frozen protocol.
5. Validate the retained mechanism on Sherlock before using the remaining five
   datasets.

## Explicitly deferred

Poetry datasets, poetry EEG/fMRI QC, and poetry experiments are paused. They do
not resume until the original CEWM/ERCM memory plan has a completed decision
report or the user explicitly reopens that track. No poetry cluster jobs or
data-acquisition work may be started under the present scope.
