# Data scope and attribution

## Synthetic world

The procedural simulator is included under [src/cmwm/synthetic](../src/cmwm/synthetic/).
It produces controlled state/action histories, entity identities, event boundaries and evaluation
targets. Versioned generator policies differ across development studies; the E1 audit counts are
not a description of a new E3 test set. Symbolic observations, not video embeddings, are the inputs
to the completed recurrent-model studies.

The record of 97,548 audited E1 queries is a data-integrity count. E2/E3 model endpoint scores each
use 512 unique validation questions, reused across seeds. Likewise, 128,000 audited E3 prediction
rows across checkpoints do not mean 128,000 independent examples.

## Sherlock

The completed analysis uses released encoding-label and recall-related products associated with
the Sherlock naturalistic-memory studies. Source hashes, aggregate metrics and development split
metadata are retained; raw transcript tokens and query-level rankings are not redistributed.
The development set has 12 participants and 401 scored queries. Five participants remain held out.
One zero-duration annotation row and a timestamp reset between source parts were retained and
explicitly audited, not silently removed.

## FilmFest

Historical analyses use released behavioral coding and source-data centralities. They are separate
from model training. Missing participant/movie entries must not be treated as observed non-recall.
The graph-retrieval diagnostic uses predicted edges, not independently annotated human directed edges.
Its graph-defined targets prevent an independent superiority claim.

## Sources and permissions

- [Janice Chen Lab dataset index](https://jchenlab.johnshopkins.edu/)
- [Sherlock dataset, OpenNeuro ds001132](https://openneuro.org/datasets/ds001132)
- [FilmFest dataset, OpenNeuro ds004042](https://openneuro.org/datasets/ds004042)
- [FilmFest event-network memory study](https://www.nature.com/articles/s41467-022-31965-2)
- [Historical dataset scope](../archive/notes/CHEN_DATA_SCOPE.md)
- [Historical metadata audit](../archive/notes/OPENNEURO_METADATA_AUDIT.md)

Links identify original providers and publications. Metadata auditing is not evidence that raw
imaging data were downloaded or modeled. This project has not validated a neural encoding or
decoding model. It does not claim to have used every dataset from the lab.

Third-party source data, media, transcripts and model weights are not licensed by this repository.
Users must verify provider permissions and cite the original studies before acquisition or reuse.
