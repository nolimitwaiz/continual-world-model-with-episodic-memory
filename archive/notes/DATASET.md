> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# CMWM-Synth-E1 Dataset Specification

## Purpose

CMWM-Synth-E1 is a procedural, deterministic dataset for testing one claim: whether grouping
experience at meaningful event boundaries improves long-delay episodic recall under a fixed storage
and retrieval budget.

It is not intended to resemble internet-scale natural data. Its purpose is to expose ground truth
that natural video cannot provide reliably: object identity, state transitions, event boundaries,
interventions, and direct action effects.

## Two synchronized observation layers

Every episode has one underlying transition trace and two possible observation layers:

1. **P1A symbolic state:** episode-randomized entity identities/slots, positions, containment,
   visibility, appearance/role metadata, agent actions, and state changes. This is the first
   experimental input.
2. **P1B rendered frames:** images rendered from the sealed P1A manifest. This is generated only
   after CMWM-E1 passes its symbolic-state gates.

The visual stage therefore changes observation quality without changing the world, queries,
boundaries, or answers.

## Episode contents

Each episode records:

- an immutable initial world state;
- the partial observation available to the agent at each step;
- the action taken at each step;
- the factual next state;
- a counterfactual next state under `do(action = wait)`;
- versioned semantic events and boundary labels;
- action-to-state causal edges supported by the factual/counterfactual difference;
- generator family, template, mechanism, seed, and split identifiers;
- structured delayed queries and exact recomputable oracle answers.

The initial mechanics cover agent motion, pickup/drop, open/close, containment, visibility changes,
and transporting a closed container with an object inside. Collision mechanics are not yet part of
the implemented generator and cannot be claimed as tested.

## Event ontology

A boundary is emitted causally after processing transition `t`. It closes and writes the semantic
segment containing that transition and its observed outcome; the next segment begins with the next
transition. Internally, the boundary is represented as the split position between those two
segments. Prototype boundaries can be caused by:

- object-relation changes such as pickup, drop, insertion, or removal;
- container-state changes;
- completion of container transport;
- visibility changes.

Ordinary motion is not automatically a boundary. This ontology is versioned because boundary
ground truth is a modeling choice, not an objective fact of nature.

## Causal annotation

The generator does not infer causality from temporal adjacency. For each action transition it also
simulates the same prior state under `do(action = wait)`. A direct causal edge is recorded only for
a state variable whose factual value differs from that counterfactual value.

These edges are evaluation labels. They are not exposed to the model, memory writer, retriever, or
query head.

Causal recall examples require matched candidate sources. Their construction must separate causal
relevance from temporal adjacency, target field, entity overlap, semantic similarity, mechanism
frequency, and presentation order. A causal query without this shortcut audit is ineligible for
LDMS.

Implemented causal queries describe a unique observed outcome by the affected state variable and
its factual value; they do not disclose the source transition. The answer is the earlier observed
action record whose intervention-supported edge produced that outcome. Audit-only candidate
sidecars provide recency-band, entity-overlap, target-field, and structural-semantic controls. The
model receives the outcome description, never the correct candidate ID or a causal edge.

## Split design

Random frame splits are forbidden. Manifests isolate seeds, templates, and mechanism compositions.

- **Train:** known pickup, drop, access, and containment compositions.
- **Validation:** disjoint seeds using known mechanism families.
- **HF-A test:** held-out persistence/action compositions with longer hidden intervals.
- **HF-B test:** held-out container-transport compositions.

The generator now uses unique seeded entity identities, randomized order-preserving event schedules,
three training templates, explicit color/shape/role fields, two held-out families, exact query
recomputation, matched causal-control sidecars, and full random/oracle segment-length evidence. After
the small engineering fixture passed, authorized CLSP job 1753878 generated and audited the full
declared 12,000-episode dataset. G0 passed with 97,548 queries, zero errors, exact oracle/random
segment-length matching, and real storage pressure. The source-bound manifest SHA-256 is
`b787ad5b1f1097de552ceb266a33506a4b8b95f6738d2eeb6fbf05b9ccf4e74c`. This validates the dataset
contract only; no model condition has been trained or compared.

Collision and long multi-event causal-chain mechanics remain absent. Collision absence is reported as
a warning and cannot support a collision claim; multi-event causal credit remains a later experiment.

## Primary data conditions

Every memory condition receives the same episode and query manifests:

1. fixed context only;
2. frame memory;
3. fixed chunks;
4. random boundaries matched to the oracle count/length distribution;
5. oracle event boundaries;
6. learned event boundaries, only if oracle boundaries pass G1.

Storage is compared in bytes, retrieval is capped at four records, and the temporal core and query
head remain identical.

## External datasets

No external dataset is required for the first claim. If CMWM-E1 passes symbolic and rendered gates,
a new decision record will select a small natural-video benchmark based on the demonstrated
capability. Ego4D-style episodic recall, controlled physical-reasoning video, and robot trajectories
remain candidates rather than commitments.
