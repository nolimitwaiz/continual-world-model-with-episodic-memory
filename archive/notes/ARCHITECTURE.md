> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# CMWM-E1 Architecture

CMWM and its event-structured episodic world memory are one integrated model.
The memory policy is exposed as an ablation boundary, not as a second model or
product: the encoder, temporal core, writer, retriever, and answer head are
shared across every comparison condition.

## 1. Architectural question

Prototype 1 tests whether event boundaries provide a useful unit for episodic storage and retrieval. The architecture is intentionally conventional everywhere else.

The causal chain under test is:

```text
continuous experience
        -> segmentation policy
        -> memory records
        -> equal-budget retrieval
        -> long-delay answer
```

The experiment must not silently change the encoder, temporal model, summarizer, retriever, query head, training data, or optimization budget when the segmentation policy changes.

## 2. Stage P1A: oracle-state system

```text
procedural world
   |-- rendered observations (recorded, not used in P1A)
   |-- oracle symbolic state s_t
   `-- observed action a_t
                |
                v
     fixed state/action adapters
                |
          fixed step fusion
                |
                v
         fixed temporal core
                |
          +-----+-------------------+
          |                         |
          v                         v
   current context            memory writer
                                    |
                              memory store
                                    |
                              fixed retriever
          |                         |
          +------------+------------+
                       v
                 fixed query head
                       |
                       v
                    answer
```

### Procedural world

The E1 world produces sequences containing objects, positions, containers, occlusion, actions, and
intervention-supported direct effects. Velocity, collision, and long causal-chain mechanisms require
later dataset versions and are not claimed by E1. Every sequence must expose:

- complete symbolic state at each step;
- entity identity and attributes;
- action and outcome labels;
- event identifiers and boundary times;
- temporal relations;
- causal annotations supplied by the simulator;
- rendered observations reserved for Stage P1B;
- generator family, mechanism, composition, and seed identifiers.

Ground-truth causal edges may be used for scoring causal-recall answers. They may not be passed to the temporal model, memory writer, retriever, or answer head in Prototype 1.

### State adapter

The state adapter and action adapter convert the same safe symbolic schema to one 128-dimensional
step vector for every condition. The action path contains only the observed action fields, never its
oracle effect or counterfactual annotation. Entity IDs are mapped through an episode-specific random
slot permutation that is stable across that episode's states, actions, and queries, independent of
split, template, outcome, and condition. A fixed global role-based ordering is forbidden.

### Temporal core

The primary temporal core is a GRU with a 128-dimensional hidden state. Its architecture, initialization policy, optimizer, and training objective remain fixed across all Prototype 1 memory conditions.

A Transformer may be reported as a robustness baseline after the primary GRU comparison. It cannot replace the preregistered GRU result or be selected post hoc because it favors one memory condition.

### Query head

The same query head answers all conditions. It receives current temporal state, the query
representation, and at most four retrieved records. Ordinary state answers use a fixed vocabulary;
causal-candidate answers use pointer scores over the retrieved-record axis. Opaque record IDs are
mapped to evaluator candidates outside the model. The head does not receive record IDs, oracle
candidate identity, causal edges, or the segmentation condition identifier.

## 3. Memory record contract

Every memory-enabled condition uses the same record payload:

```text
MemoryRecord
  key: 128 float32-equivalent values
  value: 128 float32-equivalent values
  start_step: integer
  end_step: integer
  entity_mask: fixed-width bit field
  boundary_source: hidden from the model and used only for audit
```

Keys and values are produced by the same fixed writer/summarizer. Only the observation grouping policy changes.

The writer, recurrent backbone, key/value schema, and retriever form a shared learned key/value
episodic-memory baseline across C1-C5. Prior work has already proposed recurrent models with
separate episodic keys and values; CMWM-E1 does not claim that design as novel. Its isolated variable
is which online boundary policy closes and writes a record.

The memory budget is 64 fixed-width records, or 68,096 bytes under the declared 1,064-byte layout.
Accounting includes key/value storage and metadata. The 64-record ceiling is below the 160-step
episode horizon, so frame memory experiences real FIFO eviction at the primary lags. If an
implementation uses a compressed datatype, the other conditions receive the same byte budget
rather than the same record count. A query may return at most four records.

Eviction is FIFO in Prototype 1. Learned importance, novelty, consolidation, and replay prioritization are deferred because they would add experimental variables.

## 4. Comparison conditions

### C0: fixed context only

No external memory. The model receives the latest 32 observations.

### C1: frame retrieval

Each stored record summarizes one observation. Storage, eviction, and top-k retrieval obey the common contract.

### C2: fixed chunks

Consecutive observations are grouped into chunks whose length is fixed from training-set oracle statistics before evaluation. The summarizer and retriever are identical to the event conditions.

### C3: random matched chunks

Boundaries are sampled so that the number and empirical length distribution of chunks match the oracle-event condition. Random boundaries are generated independently of task labels and outcomes.

### C4: oracle event boundaries

The simulator boundary signal controls grouping. The model receives no event label, event type, outcome, or causal edge. Oracle information changes only where a record begins and ends.

### C5: learned event boundaries

A detector predicts boundaries from information available at the current step. It may use current and prior state, action, and temporal prediction error. It may not use future frames, future outcomes, oracle event identifiers, causal annotations, or query labels.

The detector is trained only after C4 passes the oracle-event gate.

The detector emits a continuous boundary probability as well as the thresholded decision. Its
diagnostics include calibration and cue ablations for normalized predictive surprise, semantic or
object-set change, action/goal change, time gap, and latent belief/context change. Raw pixel error is
not an accepted boundary feature.

## 5. Tasks

Prototype 1 evaluates four task families:

1. **Delayed state recall:** identify an entity attribute at a specified past time.
2. **Object-location history:** report where an entity was before or after a named event.
3. **Occlusion continuity:** identify or locate an entity after it leaves the fixed context through occlusion or containment.
4. **Causal predecessor recall:** select the annotated prior event that directly produced a specified outcome among matched distractors.

Queries occur at lags of 16, 32, 64, and 128 observations. The primary score uses lags 64 and 128, both beyond the 32-observation fixed context.

Causal predecessor recall tests retrieval of simulator-defined causal information present in experience. It is not evidence that the model discovered causal structure from passive observation.

The causal-query generator must match distractors so that temporal adjacency, target field, entity
overlap, semantic similarity, or action frequency alone cannot identify the answer. These shortcut
features are audited at G0.

## 6. Stage P1B: frozen-vision transfer

P1B is conditional on the learned-event gate passing in P1A.

The oracle state adapter is replaced with one frozen DINOv3 vision encoder and a fixed projection to
the same 128-dimensional interface. The exact checkpoint, input resolution, layer/readout rule, and
license must be frozen before P1B data access. The temporal core, query head, memory contract, data
mechanisms, budgets, and comparison conditions remain unchanged.

P1B does not claim that DINOv3 is the best representation and does not train or compare CNN, ViT,
DINO-style, visual-text, or multimodal encoders. It holds one strong frozen visual interface fixed so
that G3 tests only transfer of the event-memory effect. A later representation study compares CNN,
ViT, frozen DINOv3, and an in-domain DINO-style encoder; CLIP-style alignment is a separate
multimodal study. See `RESEARCH_MAP.md`.

## 7. Controlled information flow

The following invariants are mandatory:

- no future observation enters the boundary detector or memory writer;
- no oracle event label enters learned conditions;
- no causal edge enters a predictor or retriever;
- train, development, and test generator seeds never overlap;
- held-out mechanisms and compositions cannot be reconstructed from cached training episodes;
- the query distribution is identical across memory conditions;
- the same episode instances are evaluated across paired conditions;
- memory byte limits and retrieval-count limits are enforced, not merely logged.

## 8. Deferred architecture tracks

These are explicitly outside Prototype 1:

- graph-structured storage or graph-isomorphism retrieval;
- CfC, liquid, neural ODE, JEPA, or alternative temporal cores;
- recursive or adaptive-depth computation;
- fast weights, TTT, LoRA, or online parameter updates;
- predictive coding, synthetic gradients, feedback alignment, Forward-Forward, NoProp, or learned update rules;
- pixel-language and multilingual orthographic representations;
- audio and shared/private multimodal representations;
- learned eviction, replay, consolidation, or slow/fast weight arbitration;
- data-quality scoring, curriculum learning, and mixture optimization;
- natural-video pretraining, robotics, and models at foundation scale.

A deferred track requires a new decision record stating its hypothesis, simpler baseline, budget, falsification gate, and why Prototype 1 evidence justifies it.

The ordering and named role of every deferred component are binding in `RESEARCH_MAP.md`.
