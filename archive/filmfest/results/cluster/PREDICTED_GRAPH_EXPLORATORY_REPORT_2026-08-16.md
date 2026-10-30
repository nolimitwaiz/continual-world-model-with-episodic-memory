> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# FilmFest predicted-graph exploratory retrieval report

Date: 2026-08-16  
Remote root: `<CLUSTER_PATH>`  
Scientific status: **exploratory, circular predicted-graph diagnostic**  
Training performed: **no**

## Cluster execution

| Job | Purpose | Result |
| ---: | --- | --- |
| `1753889` | benchmark path/label safety tests | 8/8 passed |
| `1753890` | Narratives GraphQL task audit | completed; zero-payload result rejected as incomplete |
| `1753891` | first MPNet manifest attempt | failed before model load because released end-time was blank; no artifact |
| `1753892` | Narratives git-annex pointer audit | completed without payload fetch |
| `1753893` | corrected predicted-graph manifest + frozen MPNet inference | completed on A100 |
| `1753894` | post-cache benchmark safety tests | 8/8 passed |
| `1753895` | BM25, frozen E5, and predicted-causal retrieval at K=3/5/10 | completed on A100 |
| `1753896` | first complete-suite staging check | 24 passed; 3 fixture errors because `results/` was excluded from staging |
| `1753897` | complete suite after restoring fixed reference fixtures | 27/27 passed |

The corrected graph uses the released discrete `event_index` as its complete
time coordinate. It does not impute missing wall-clock seconds.
Job `1753896` did not expose a code failure; its three errors were missing fixed
reference JSON files in the fresh transfer root. Those exact fixtures were
restored without alteration before job `1753897` passed the full suite.

## Manifest

- 202 FilmFest events, 10 stories, and 575 retained Qwen-predicted directed edges;
- 386 base questions: 143 local, 129 medium, and 114 long;
- 1,930 rows over five fixed outer folds;
- 52 out-of-fold test questions in the long-distance/low-overlap target cell,
  spanning eight stories;
- overlap encoder: `sentence-transformers/all-mpnet-base-v2` at immutable
  revision `e8c3b32…`, 768 dimensions, attention-masked mean pooling and L2
  normalization.

Question selection precedes overlap scoring. The target-cell coverage check
passes for this exploratory manifest.

## Retrieval result at K=5

| Method | Graph input | Local/high CPR | Long/low CPR |
| --- | --- | ---: | ---: |
| BM25 | none | 0.833 | 0.410 |
| frozen E5 | none | 0.879 | 0.487 |
| predicted causal | Qwen predicted graph | 1.000 | 0.744 |

For frozen E5, `long/low - local/high = -0.3916`; the 10,000-resample
story-cluster bootstrap 95% interval is `[-0.4938, -0.2470]`. In the target
cell, predicted-causal minus E5 is `+0.2564`, interval `[+0.1014, +0.3607]`.

## Required interpretation

This establishes an engineeringly coherent failure map: text similarity is
substantially weaker when the selected causal path is long and its earliest
cause has low semantic overlap with the anchor. It does **not** establish an
independent graph-memory advantage. The same Qwen-predicted graph defines the
required paths and powers `predicted_causal`, and Qwen labeled all stories
rather than learning an edge predictor only from each fold's training stories.

Consequently:

- `predicted_causal` is never renamed `oracle_causal`;
- confirmatory Gate B remains blocked;
- ERCM is not validated by these numbers;
- the next scientific requirement is an independent human directed graph (or
  a separately preregistered train-only edge-prediction study), followed by
  Sherlock external validation.

Returned artifacts and SHA-256 records are in
`results/cluster/predicted_manifest_v2_1753893/` and
`results/cluster/predicted_retrieval_v2_1753895/`.
