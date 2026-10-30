> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# CMWM Design Audit — Event-Structured Episodic World Memory

Date: 2026-08-10

Status: research audit, not an empirical result.

## 1. Executive verdict

The pasted design is coherent and worth testing, but its broadest novelty claim is not defensible.
Event-segmented memory, surprise-gated writing, episodic/semantic memory graphs, and episodic world
models all have close prior art. The program remains potentially distinctive if it makes and tests
a narrower claim:

> Under matched storage, retrieval, parameter, data, and compute budgets, can causal online event
> segmentation and learned experience structure improve long-delay memory first, then continual
> adaptation and credit assignment, without privileged graph labels?

The current CMWM-E1 protocol tests only the first clause. It must not claim continual learning,
causal credit assignment, reinstatement, graph reasoning, or superiority to RAG until those are
tested separately.

## 2. Exact conceptual inventory

The pasted text contains the following ideas. They are separated because each needs its own
baseline and can fail independently.

| ID | Proposed idea | Scientific role | Earliest legitimate test | Decision now |
| --- | --- | --- | --- | --- |
| I01 | Separate reference memory from experience memory | Prevent one retrieval mechanism from serving incompatible jobs | RAG versus episodic memory on matched static and temporal tasks | Keep as program architecture |
| I02 | Store event abstractions rather than raw frames | Compression and invariance | Frame, fixed-chunk, random-matched, oracle-event records | CMWM-E1 core |
| I03 | Detect internal as well as sensory boundaries | Segment goal/context changes with little pixel change | Online cue ablations with matched boundary rate | G2 diagnostic, mechanism later |
| I04 | Use calibrated soft boundary probability | Avoid brittle threshold-only segmentation | F1, AUPRC, Brier score, ECE, and LDMS | G2 diagnostic |
| I05 | Hierarchical micro/event/episode boundaries | Represent multiple timescales | Multiscale synthetic task with scale-specific queries | Defer; one scale first |
| I06 | Event-level latent reinstatement | Restore task-relevant prior state, not just text | Retrieval-conditioned latent reconstruction plus downstream utility | Post-G3 |
| I07 | Event graph with temporal, semantic, causal, entity, agent, place, goal, and outcome edges | Structured retrieval and analogy | Learned graph versus vector retrieval with oracle edges forbidden | R4 |
| I08 | Centrality-aware retention/replay | Prioritize structurally important events | Equal-byte FIFO versus learned centrality policy | R4/R5 |
| I09 | Agency-aware memory | Distinguish observed from self-caused outcomes | Matched active/passive trajectories | R4/R7 |
| I10 | Boundary-triggered reactivation | Read related memories at transitions | Equal-read boundary, periodic, always-on, and random schedules | E2 after G2 |
| I11 | Persistent latent world state | Track current hidden state through partial observation | Oracle-state GRU first, pixel transfer later | CMWM-E1 backbone |
| I12 | Prediction surprise as one boundary/write/replay signal | Couple prediction and memory | Cue ablation with uncertainty normalization | Diagnostic only at first |
| I13 | Goal relevance, value, novelty, uncertainty, redundancy, regime, causality, and agency as memory controls | Avoid surprise-only memory | One feature at a time against uniform/FIFO | R5/R7 |
| I14 | DINO-like visual representation | Dense visual state interface | Frozen DINOv3 transfer after symbolic gates | G3 |
| I15 | CLIP-style multimodal alignment | Image-language semantic grounding | Independent encoders/late fusion versus CLIP-style alignment | R3 |
| I16 | Shared/private multimodal factors | Preserve aligned and modality-specific information | Non-collapse probes and downstream gain | R3 after CLIP baseline |
| I17 | Visual orthography | Generalize across scripts/noisy text | Token/character versus rendered-text encoder | R3 |
| I18 | GRU/Transformer/JEPA/CfC dynamics | Predict controlled temporal state | One dynamics comparison per protocol | GRU now; others gated |
| I19 | Recursive reasoning with adaptive depth | Allocate compute by difficulty | Fixed 1/2/4/8 steps before learned stopping | Later controlled-depth task |
| I20 | Working, episodic, fast-weight, and slow-weight timescales | Separate immediate state, experience, adaptation, and knowledge | Staged ablations | Program thesis |
| I21 | TTT/fast weights/online LoRA | Fast within-regime adaptation | 1/2/5/10-shot adaptation and retention | R5 |
| I22 | Evidence-weighted consolidation | Convert repeated episodes into slow knowledge selectively | Uniform replay and reservoir/FIFO controls | R5 |
| I23 | Graph structural analogy | Transfer causal form across different entities | Held-out isomorphic and near-isomorphic mechanisms | R4 |
| I24 | Event/causal structure for temporal credit assignment | Restrict which past events receive outcome credit | Sparse delayed-reward benchmark against TD/retrace/chunked controls | Separate post-R4 protocol |
| I25 | Learned local update rule informed by memory and causal evidence | Reduce global backpropagation | Small frozen benchmark with true-gradient teacher | R6 only |
| I26 | Predictive coding, equilibrium propagation, synthetic/forward gradients, Forward-Forward, and NoProp | Alternative learning controls | Separate controlled algorithm studies | R6; never stack together |
| I27 | Data selection by quality/novelty/redundancy/uncertainty/causal relevance | Improve sample efficiency | Stable-model data-mixture experiment | R7 |
| I28 | Regime-shift continual-learning environment | Measure adaptation and forgetting | Blue-fragile/durable/material-like controlled rules | R5 |
| I29 | RAG plus episodic memory in one system | Combine documents with lived interaction | Four-way ablation: neither, RAG, episodic, both | After each memory works alone |
| I30 | Full multimodal continual world model | Integration endpoint | Factorial integration of independently passing parts | Scale stage only |

## 3. What the Chen-lineage evidence does and does not support

The Chen Lab work is useful for hypotheses about event organization, internally generated
boundaries, event-network structure, agency, and reinstatement. It is observational human-memory
evidence, not an AI algorithm and not proof that any CMWM module will work.

Supported as motivation:

- event boundaries and high-level event representations;
- internally generated context shifts without large sensory changes;
- semantic and causal network structure as predictors of later memory;
- partial reinstatement of experience-related patterns during recall;
- agency changing the organization of remembered experience.

Not supported as an engineering conclusion:

- that an event gate beats fixed chunking under an equal byte budget;
- that fMRI pattern reinstatement implies reconstructing a learned world-model latent;
- that human-coded narrative causal links may be exposed to the model;
- that causal predecessor recall solves reinforcement-learning credit assignment;
- that the resulting system is biologically equivalent or generally better than RAG.

Primary Chen Lab publication map: <https://jchenlab.johnshopkins.edu/>.

## 4. Novelty audit

The general area is active and increasingly crowded. At minimum, any paper or proposal must compare
against or distinguish itself from the following primary work:

| Work | Existing overlap with our design | Remaining distinction to test |
| --- | --- | --- |
| [EM-LLM](https://arxiv.org/abs/2407.09450) | Bayesian-surprise event segmentation, graph refinement, episodic retrieval, temporal contiguity, comparison with RAG | CMWM uses embodied/symbolic world state, intervention-backed evaluation, equal storage/compute controls, internal action/goal cues, and later continual learning |
| [AriGraph](https://arxiv.org/abs/2407.04363) | Semantic plus episodic graph world model for agents | CMWM must learn graph structure without LLM extraction or oracle causal edges and test matched vector/graph budgets |
| [REMem](https://arxiv.org/abs/2602.13530) | Hybrid time-aware memory graph and multi-step episodic reasoning | CMWM targets causal online experience, latent state, and continual-learning effects rather than offline text indexing alone |
| [HiMem](https://arxiv.org/abs/2601.06377) | Hierarchical episode/note memory, event-surprise segmentation, reconsolidation | CMWM needs controlled causal tasks and matched intervention tests rather than dialogue-only gains |
| [Worth Remembering](https://arxiv.org/abs/2606.03787) | Surprise-gated robot episodic memory with V-JEPA-2 and 4D scene graphs | CMWM cannot claim surprise-gated robot memory as novel; it can test richer internal cues, agency, and equal-budget learning effects |
| [World Model Driven Episodic Memory for LLMs](https://openreview.net/forum?id=zxYTR2UBG2) | Actor/role/state representations across space and time; reported advantage over standard RAG | CMWM must establish causal online learning and model-independent controlled results |
| [Sequence Compression Speeds Up Credit Assignment](https://proceedings.mlr.press/v235/ramesh24b.html) | Learned sequence chunking used to improve temporal credit assignment | Any CMWM credit claim must compare with Chunked-TD and standard TD/trace baselines |
| [Graph-Based Continual Learning](https://arxiv.org/abs/2007.04813) | Graph-augmented episodic replay for continual learning | CMWM must isolate event/causal structure rather than graph similarity alone |

Therefore the phrases “nobody has thought of this” and “fundamentally beyond RAG” are not acceptable
scientific claims. A viable novelty statement is conditional and narrower:

> CMWM studies whether causally online, internally cued event segmentation improves bounded
> experience memory and whether learned event structure can subsequently route replay and temporal
> credit in controlled world-model settings.

That statement is still a hypothesis until the relevant stages pass.

## 5. Highest-risk failure modes

1. **Oracle leakage.** Simulator event IDs, causal edges, counterfactual states, query answers, and
   future outcomes must never enter model-visible features.
2. **Boundary-rate confounding.** Event methods can win merely by writing fewer or longer records.
   Random controls must match the full segment-length distribution and byte occupancy.
3. **Reference-step leakage.** A causal query can become trivial if the disclosed time or target
   field identifies the answer.
4. **Entity/template shortcuts.** Fixed IDs, deterministic schedules, or small template sets can
   replace memory with metadata classification.
5. **Fake storage pressure.** Long-delay questions must reference observations actually evicted from
   both context and FIFO memory.
6. **Semantic similarity masquerading as causality.** Recency, entity overlap, target field, and
   semantic similarity must be independently matched.
7. **Surprise-only oversegmentation.** Stochastic visual nuisance can generate false boundaries;
   use uncertainty-normalized surprise and matched boundary rates.
8. **Expected but important events.** Goal/value relevance must eventually be evaluated separately
   from prediction error.
9. **Pseudo-replication.** Thousands of episodes from one trained seed are not thousands of model
   replications.
10. **Kitchen-sink rescue.** DINO, CLIP, JEPA, graph memory, TTT, or CfC may not be added to rescue a
    failed E1 test.
11. **Credit-assignment overclaim.** Retrieving a causal predecessor is not equivalent to improving
    a learned policy under delayed reward.
12. **Premature cluster scaling.** A code path that runs is not evidence that a mechanism works.

## 6. Correct experiment ladder

1. **G0 — dataset validity.** Generate, seal, and audit diverse CMWM-Synth-E1 manifests. Block all
   model comparisons on leakage, shortcuts, missing causal controls, or inadequate storage pressure.
2. **Engineering smoke matrix.** Exercise data, tensors, each allowed memory grouping, backward
   pass, checkpoint, determinism, and artifact logging on CUDA. These runs establish only that code
   executes.
3. **C0-C3 baselines.** Fixed context, frame memory, fixed chunks, and random matched chunks.
4. **G1 oracle-event test.** Compare oracle events only with the development-selected strongest
   non-event baseline under matched budgets. Stop the event branch on failure.
5. **G2 learned online boundaries.** Test only after G1 passes. Report boundary calibration and cue
   ablations. One bounded revision is allowed.
6. **G3 rendered visual transfer.** Freeze one DINOv3 interface and repeat the memory comparison.
7. **E2 read timing.** Fix the winning writer; test boundary-triggered reactivation against
   equal-read schedules.
8. **R4 learned structural memory.** Compare vector and graph retrieval with no oracle edges.
9. **R5 continual learning.** Add regime shifts, then compare context, retrieval, replay, and fast
   adaptation.
10. **Credit assignment.** Only after learned causal structure is trustworthy, test whether it
    improves delayed-reward learning against standard RL and sequence-compression baselines.
11. **Integration.** Combine only components with independent positive evidence and test their
    interaction factorially.

## 7. What is built versus merely proposed

Built:

- symbolic schemas, mechanics, events, one-step counterfactual causal labels, and exact queries;
- event segmentation utilities and gate arithmetic;
- G0 manifest, leakage, shortcut, storage-pressure, and confound audits;
- tensor contracts for state/action encoding, GRU, key/value writer, and query head;
- fail-closed CUDA/Slurm and submission guards;
- condition configurations and scheduler templates.

Not built:

- a generator version that passes G0;
- stable randomized entity-slot collation and frozen answer/query vocabularies;
- a complete end-to-end model and training/evaluation loop;
- learned online boundary detector training;
- rendered frames or DINO integration;
- graph memory, reinstatement, fast weights, consolidation, or continual learning;
- an authenticated cluster transfer or any result.

## 8. Naming

Use **CMWM** (Continual Multimodal World Model) for the umbrella program and **CMWM-E1** for the
current falsification study. Use **Event-Structured Episodic World Memory** as a descriptive
subsystem name, not as a novelty claim. A permanent product or paper name should wait until at
least G1 passes and a trademark/domain search is performed.

