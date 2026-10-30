> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# GI-1 draft: equal-budget structural episodic retrieval

Status: draft only; not authorized to run  
Dependency: the normal CMWM-E1 gate sequence and a separate run decision  
Model affected now: none

## Question

When the event-writing policy, recurrent state, memory bytes, read count, and
downstream head are fixed, does structure-aware retrieval improve long-delay
causal transfer under entity and appearance changes relative to ordinary
content-addressed retrieval?

The null is that it provides no improvement at equal budget, or that gains are
offset by false matches, abstention, or latency.

## Non-negotiable information boundary

Simulator counterfactual causal edges, template IDs, held-out-family labels,
future states, and answers are scoring data only. They cannot be stored,
queried, or used to construct a model-time graph.

The retrieval graph must be derived solely from inputs already available to
every condition. In the symbolic benchmark these may include the observed
action type, observed entity kind/slot, event membership, and observed or
model-predicted state-field changes. If a field is unavailable to the content
baseline, it is unavailable to the structural condition.

## Frozen components

- encoder and recurrent dynamics;
- event boundary/write policy;
- memory record layout and total 68,096-byte capacity;
- maximum of four retrieved records per query;
- query point, downstream prediction head, optimizer, seed set, and train data;
- permitted evidence and local-context window.

Any bytes used for cached WL colors, hashes, adjacency, or labels count against
the memory budget. Prefer deriving the query fingerprint and stored-record
motif from already budgeted fields so capacity does not silently increase.

## Conditions

1. **C-content:** existing content-addressed retrieval.
2. **C-structure:** directed edge-labeled 1-WL candidate indexing followed by
   bounded exact label-preserving verification; abstain when no verified motif
   exists.
3. **C-random:** random retrieval matched to C-structure's realized read count
   and abstention rate. This separates structural matching from simply reading
   memory more or less often.

An optional hybrid is not part of the first comparison. If later tested, it
must use the same content score only as a tie-break inside the verified
structural candidate set and receive its own decision row.

## Dataset perturbations

Use a fresh, sealed exploratory manifest rather than the GI-0 fixture or the
confirmatory E1 test set. Each causal program should have paired surface forms:

- bijective entity-slot renaming;
- color, shape, and role changes;
- position and schedule changes that preserve causal order;
- hard negatives sharing action/entity-label histograms but differing in edge
  direction, causal mechanism, or action order;
- genuinely novel motifs requiring abstention.

The generator must include more than one graph per template and more than one
template per easy action histogram so retrieval cannot reduce to template or
bag-of-actions classification.

## Metrics and gates

Primary model metric: the same long-delay memory score used by the applicable
CMWM gate, reported separately for renamed-equivalent, hard-negative, and
novel-motif strata.

Structural diagnostics:

- retrieval precision and recall against ground-truth relevant episodes;
- false-match and abstention rates;
- exact whole-graph versus partial-motif match counts;
- performance conditioned on motif frequency and lag;
- retrieval wall time, p50/p95 latency, nodes visited, and exact-search states;
- peak memory and total stored bytes.

Proceed beyond an exploratory pilot only if C-structure beats C-content on the
predeclared primary held-out stratum across seeds, does not regress the ordinary
in-distribution stratum beyond the declared tolerance, stays within the byte
and read budgets, and has no unbounded-search event. Stop if graph construction
uses oracle labels or if a simpler action-sequence baseline explains the gain.

## What GI-1 would and would not establish

A positive result would show that explicit event structure helps retrieve
causal analogies under surface change. It would not establish learned causal
discovery, natural-video graph quality, continual consolidation, or a benefit
from graph neural networks. A GIN or other learned graph encoder is a separate
follow-up only after the exact symbolic baseline earns its complexity.
