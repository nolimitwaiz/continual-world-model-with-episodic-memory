> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# GI-0: Graph-isomorphism study for CMWM

Decision date: 2026-08-21  
Status: exploratory structural audit; not a Prototype 1 memory condition

## Why it may help

CMWM should recognize that two episodes can instantiate the same causal
structure even when entity IDs, colors, positions, and language differ. Graph
isomorphism provides an explicit test of that invariance. It is most useful for:

1. canonicalizing structurally equivalent episodic graphs;
2. retrieving prior causal motifs under entity renaming;
3. measuring whether held-out causal compositions reuse known substructures;
4. testing whether a later graph encoder preserves distinctions that exact
   structure can make.

It does not solve event segmentation, causal-edge discovery, memory writing,
or continual learning. Exact matching also cannot recognize a genuinely new
topology as equivalent to an old one.

## GI-0 representation

Each synthetic episode becomes a directed labeled bipartite graph:

```text
action(type) --causal:mechanism--> state-effect(entity-kind, field)
     |
     +--temporal:next--> next causal action
```

Random entity IDs and state values are excluded. Action type, causal mechanism,
entity kind, affected field, edge direction, and causal-action order remain.

The implementation uses four rounds of directed, edge-labeled 1-WL refinement
as a fast permutation-invariant index. Candidate matches are then checked by a
bounded exact label-preserving isomorphism verifier. A WL collision alone is
never called an exact match.

## First falsifiable questions

- **GI0-A invariance:** Episodes from the same generator template retain one
  verified structural identity across randomized entity names and schedules.
- **GI0-B separation:** Distinct templates are not merged by the verified
  matcher in the audit fixture.
- **GI0-C compositional coverage:** Three-action motifs from HF-A/HF-B are
  compared with the training motif vocabulary. WL retrieves candidates and
  exact label-preserving isomorphism verifies every counted match. Candidate
  false positives and exact coverage are reported; neither is optimized.

Failure of GI0-A rejects the representation. Failure of GI0-B requires richer
labels or a stronger exact representation. Low GI0-C means exact/motif lookup
is unlikely to transfer to the held-out compositions and should not be added to
CMWM's event-structured episodic memory.

## Next decision gate

If GI-0 passes, preregister GI-1 before model training:

- baseline: existing content-addressed retrieval;
- candidate: WL-indexed, exact-verified causal-motif retrieval;
- same stored bytes, top-k, query evidence, and latency budget;
- test families must rename entities and alter surface features;
- separately report exact whole-graph matches, partial motif matches, false
  matches, abstentions, and retrieval latency.

A learned GIN is considered only after the symbolic exact/motif baseline is
useful. GIN is bounded by the discriminative power of 1-WL under its standard
message-passing construction, so it is not automatically more reliable than
exact verification.
