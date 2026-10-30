> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# CMWM Master Research Map

This document is the authoritative map from the original research vision to staged,
falsifiable experiments. A component listed as **later** is retained in the program; it is not
authorized for implementation in CMWM-E1 until its entry gate passes and a decision record freezes
its protocol.

## 1. Program thesis and present experiment

The tentative program thesis is that a useful continual multimodal world model may require four
different kinds of state operating at different timescales:

| State | Symbol | Timescale | Intended role | First direct test |
| --- | --- | --- | --- | --- |
| Working state | `h_t` | steps | Current computation and local context | CMWM-E1 |
| Episodic memory | `E` | written immediately | Specific events that can be retrieved later | CMWM-E1 |
| Fast weights | `W_t` | within an episode | Rapid adaptation to a new rule or entity | Track R5 |
| Slow weights | `theta` | across training | Stable statistical knowledge | Present in every trained baseline |

CMWM-E1 tests only the first two. It asks whether **where experience is segmented and written**
matters under equal byte, retrieval, parameter, data, and compute budgets. It does not yet test the
entire thesis.

Complementary Learning Systems is a conceptual inspiration for the separation between rapid
episode storage and slower statistical learning. It is not evidence that the CMWM modules are
biologically equivalent. Replay or consolidation from `E` into `theta` is an R5 hypothesis that
must beat no-consolidation and ordinary rehearsal controls.

## 2. Chen-lab inspiration: exact role and limits

The Janice Chen lab's work is a direct motivation for the event-memory branch, but neuroscience
findings are not treated as an implemented algorithm or as proof that an AI mechanism will work.
The translation is deliberately explicit:

| Finding or idea | Engineering hypothesis | CMWM test | Status |
| --- | --- | --- | --- |
| Relevant past information can be selectively reactivated at event boundaries | Boundaries may be useful times to close/write an episode, and separately may cue retrieval of related prior episodes | E1 compares fixed, random-matched, oracle, and causal learned write boundaries; E2 later tests boundary-triggered reactivation | **Write/grouping in E1; reactivation in E2** |
| Boundary responses and event changes can be graded, and meaningful internal context changes need not follow large sensory changes | A calibrated soft boundary score and latent-belief change may be more useful than raw pixel/latent error | Report boundary probability calibration and cue ablations in G2; test soft/multiscale writing only in a later protocol | **Diagnostics in E1; mechanism later** |
| Event-specific patterns can be reinstated beyond the original sensory details | Event-level records may support retrieval across visual variation | Test representation similarity only as a diagnostic; require downstream recall/transfer for a gate | **E1/P1B diagnostic** |
| Semantically and causally central narrative events tend to be better remembered; hippocampal boundary responses relate to later memory | Event-network centrality might improve later replay, consolidation, or eviction | Compare equal-budget FIFO/vector memory with centrality-aware memory after graphs are learned without oracle leakage | **R4 later** |
| Causal and chronological structure can organize recall differently from presentation order or semantic similarity | Directed event structure may retrieve causal predecessors when surface, semantic, and temporal cues are controlled | Orthogonalize recency, entity overlap, semantics, and intervention-supported causal relevance; later compare incoming- versus outgoing-edge centrality | **G0 control; R4 mechanism** |
| Narratives have causal structure that may help organize credit across long timescales | Retrieved causal event chains might route temporal evidence better than surface similarity | Structural-analogy and delayed-causal-recall benchmark against vector retrieval | **R4 later** |
| Agency changes how experiences are organized in memory | Agent-caused transitions may deserve a distinct, learned boundary/retrieval feature | Pre-registered agency ablation on interactive trajectories | **R4/R7 later** |

CMWM-E1 operationalizes only **event-based grouping and writing**. It does not yet implement
boundary-triggered reactivation, boundary-biased encoding strength, event-network centrality, or a
causal credit-assignment mechanism; those are separate experiments rather than assumed benefits.

The current detector may observe only information available at time `t`: current/prior state,
action change, object-set change, uncertainty, and prediction surprise. It cannot see future
outcomes, event IDs, causal annotations, or the later query. Raw prediction error alone is not the
final detector: stochastic nuisance changes can produce false boundaries, so the learned detector
will be compared with restricted cue ablations and calibrated for boundary rate.

Relevant primary sources include:

- [The human brain reactivates context-specific past information at event boundaries](https://pubmed.ncbi.nlm.nih.gov/37248340/)
- [Discovering event structure in continuous narrative perception and memory](https://doi.org/10.1016/j.neuron.2017.06.041)
- [Major shifts in narrative understanding can occur without major sensory change](https://doi.org/10.7554/eLife.73693)
- [Predicting memory from the network structure of naturalistic events](https://www.nature.com/articles/s41467-022-31965-2)
- [Causal structure organizes human memory of nonlinear narratives](https://doi.org/10.1162/jocn_a_02216)
- [The causal structure and computational value of narratives](https://pubmed.ncbi.nlm.nih.gov/38734531/)
- [Interactive narratives reveal the personalizing effect of agency on episodic memory](https://pubmed.ncbi.nlm.nih.gov/42251041/)
- [Janice Chen Lab](https://jchenlab.johnshopkins.edu/)

## 3. Perception and representation map

DINO and CLIP are both retained, but they answer different scientific questions.

| Component | Question it answers | Simpler control | Entry gate | Stop/keep rule |
| --- | --- | --- | --- | --- |
| Oracle symbolic state | Does event memory have value when perception is correct? | None; this is the deconfounded substrate | Start | Required for G1/G2 |
| Small CNN | How much can a task-trained conventional visual encoder recover? | Oracle state and matched projection | E1/G3 decision report complete | Keep only if it supports stable tracking/probes at the declared compute |
| Small ViT | Does patch attention improve the same visual probes over a CNN? | Small CNN | R2 representation protocol opens | Keep only with paired improvement or a clear efficiency advantage |
| Frozen DINOv3 | Do strong self-supervised dense features preserve enough object/state information for event memory to transfer to pixels? | Same episodes, temporal core, and memory conditions; representation frozen | G2 passes | **First P1B visual interface.** G3 requires positive event-memory transfer and at least 50% of symbolic uplift retained |
| Small DINO-style encoder | Does in-domain self-supervision beat transfer features on controlled data? | Small CNN/ViT and frozen DINOv3 | R2 supervised baselines established | Reject if benefit disappears under matched data/compute |
| CLIP-style alignment | Does image-language alignment create a useful shared semantic space for grounding? | Independently trained unimodal encoders plus late fusion | Visual memory works and paired image-text task is frozen | Keep only if grounding/transfer improves without damaging modality-private probes |
| Shared/private decomposition | Can aligned semantics coexist with information unique to each modality? | One shared CLIP-style vector | CLIP baseline established | Require non-collapse probes, per-modality reconstruction/variance, and a useful downstream gain |
| I-JEPA-style image objective | Does masked latent target prediction improve static visual state representations? | Same-backbone supervised and DINO-style objectives on matched controlled data | Representation track opens | Keep only on state/tracking probes and downstream LDMS under matched data/compute |
| V-JEPA-style video prediction | Does predicting future video representations improve temporal/world dynamics? | Per-frame DINO plus GRU/Transformer using the same clips | Representation and memory interfaces stable | Keep only on multi-step and occlusion rollout gates under matched compute |
| V-JEPA 2-AC-style action conditioning | Does action-conditioned latent prediction improve controllable prediction/planning? | Best passive video predictor plus the same actions | An interactive action benchmark is frozen | Keep only on held-out action-conditioned prediction/planning; not an E1 component |

The Stage P1B choice is now concrete: use one **frozen DINOv3** encoder and a fixed projection to
the 128-dimensional CMWM-E1 interface. P1B does not claim DINO is best; keeping it fixed isolates
whether the memory effect survives a pixel interface. The later representation bake-off compares
CNN, ViT, frozen DINOv3, and an in-domain DINO-style encoder without changing the downstream
system.

CLIP is not substituted for DINO in P1B. DINO supplies vision-only dense features; CLIP introduces
paired language supervision and tests cross-modal semantic alignment. Mixing them at G3 would make
failure or success uninterpretable.

If a shared/private baseline collapses private features, orthogonality, covariance reduction,
mutual-information penalties, or explicit variance/reconstruction constraints may be compared one
at a time. Low cosine similarity alone does not prove that either subspace retains useful
information.

Primary references:

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf)
- [DINOv3](https://arxiv.org/abs/2508.10104)
- [I-JEPA](https://ai.meta.com/research/publications/self-supervised-learning-from-images-with-a-joint-embedding-predictive-architecture/)
- [V-JEPA](https://ai.meta.com/research/publications/revisiting-feature-prediction-for-learning-visual-representations-from-video/)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)

## 4. Language and multimodality map

| Track | Baseline | Intervention | Required tests | Status |
| --- | --- | --- | --- | --- |
| Symbolic text | Character/subword embeddings and a small text encoder | None | Meaning, noise, and unseen-symbol controls | R3 baseline |
| Visual orthography | Same strings rendered as pixels | Salesky-style patch encoder | Latin, Arabic, Devanagari, and synthetic unseen glyphs; control semantics and rendering artifacts | R3 later |
| Hybrid text | Symbolic plus visual-orthographic features | Fixed late fusion first | Benefit by script/noise regime under matched capacity | R3 later |
| Image-text grounding | Separate encoders | CLIP-style contrastive alignment | Retrieval, grounding, compositional negatives, private-information probes | R3 later |
| Audio | Text/image system | Frozen audio interface, then aligned audio | Cross-modal retrieval and modality-private probes | R7 later |

The visual-text hypothesis comes from pixel-based multilingual modeling, not from an assumption that
language is fundamentally vision. It must beat ordinary symbolic text on a declared regime rather
than merely work. Primary reference: [Pixel Representations for Multilingual Translation and Data-Efficient Cross-Lingual Transfer](https://aclanthology.org/2023.emnlp-main.854/).

## 5. Temporal dynamics and reasoning map

| Component | Scientific question | Mandatory comparison | Entry condition |
| --- | --- | --- | --- |
| GRU | Conventional fixed recurrent baseline | Fixed context and memory conditions | CMWM-E1 now |
| Transformer | Robustness to temporal-core choice | Same interface, data, and memory; matched or reported compute | After primary GRU result |
| CfC/liquid dynamics | Does explicit continuous-time handling help irregular observations? | GRU/Transformer with `delta_t` access | Only after a benchmark includes variable frame rate, dropped frames, and asynchronous observations |
| V-JEPA-style video predictor | Does latent video prediction improve long-horizon state modeling? | Per-frame DINO plus GRU/Transformer next-state and multi-step prediction | R2/R3 interfaces stable |
| Weight-tied recursion | Does iterative refinement improve reasoning per parameter? | One pass and fixed 2/4/8 iterations | A task family with controlled required depth exists |
| Adaptive stopping | Can extra computation be allocated only when useful? | Best fixed-depth recursive baseline | Fixed-depth recursion shows value first |

CfC will not be tested on a fixed-`delta_t` dataset and then declared ineffective. Recursion will not
be tested on tasks that a one-pass shortcut can solve.

## 6. Memory and continual-learning map

| Layer | First baseline | Candidate intervention | Status |
| --- | --- | --- | --- |
| Working memory | GRU state plus fixed context | Transformer context robustness check | E1 now / robustness later |
| Episodic memory | Shared recurrent key/value writer, equal-byte FIFO vector records, and fixed retrieval | Event-gated record boundaries | E1 now |
| Boundary-triggered reactivation | Periodic, always-on budget-capped, and random-matched read schedules | Reads triggered by detected boundaries | E2 after G2 |
| Structural memory | Vector retrieval | Learned event graphs, WL-style features, subgraph/structure retrieval | GI-0 passed; GI-1 protocol only, retrieval remains R4 later |
| Fast adaptation | Context-only and retrieval | Fast weights, TTT inner updates, online LoRA | R5 later |
| Consolidation | No consolidation/FIFO | Replay into slow weights, importance/novelty-aware eviction | R5 later |

Graph memory must use information inferred from observations. Simulator causal edges remain scoring
labels and cannot be inserted into the stored graph. Exact graph isomorphism is not required; the
first question is whether structural retrieval improves held-out causal analogies whose entity
names and appearances differ.

GI-0 was a narrow exception for auditing the *ground-truth benchmark structure*: it canonicalizes
simulator causal graphs under entity renaming, indexes them with directed edge-labeled 1-WL, and
exactly verifies candidate matches. It never exposes those graphs to a model. This audit determines
whether a later structural-retrieval protocol is even coherent; it is not evidence that graph
memory works. GI-0 passed on CLSP job 1758049 with five template-pure exact classes and 55.1515%
mean per-episode exact-verified held-out motif coverage. GI-1 remains protocol-only until its own
decision gate.

The key/value recurrent memory is a shared baseline/backbone across C1-C5, informed by prior
key/value episodic-memory work. Our E1 variable is segmentation, not the existence of learned keys
and values. A post-G2 E2 protocol holds the successful write policy fixed and compares
boundary-triggered, periodic, always-on budget-capped, and random-matched read schedules with the
same number of reads, retrieved records, and inference FLOPs. This isolates Chen-inspired
reactivation timing from Chen-inspired event writing. Relevant prior art:
[A neural network model of differentiation and integration of episodic memories](https://doi.org/10.1101/2025.09.01.673596).

TTT is classified as a fast-weight mechanism, not a replacement for the learning-rule program. It
changes when an internal model adapts, but standard TTT still uses gradients for its inner update
and backpropagation for outer training.

## 7. Learning-rule map

Backpropagation remains the control. Alternative methods enter only on a small, frozen benchmark
after the model/data/evaluation platform is trustworthy.

1. Backpropagation.
2. Synthetic gradients.
3. Feedback alignment.
4. Forward gradients.
5. Predictive-coding-inspired learning.
6. Forward-Forward.
7. NoProp where its assumptions apply.
8. Local updates plus periodic backpropagation.
9. A learned update function `U_phi`, initially trained against true backprop gradients and judged
   by the loss change caused by applying its updates—not cosine similarity alone.

Every comparison reports quality against total FLOPs, wall time, peak memory, examples, and where
possible energy; convergence-to-equal-quality matters more than cost per update. The learned-rule
track includes 100%, 50%, 10%, 1%, and 0% global-correction schedules only after the full-BP
teacher baseline is stable.

## 8. Data-intelligence track

Data selection is retained as a later experimental layer, not embedded in E1 generation or used to
rescue a weak architecture. Begin with uniform random sampling and compare one intervention at a
time:

- quality filtering;
- novelty sampling;
- redundancy reduction;
- uncertainty sampling;
- difficulty curriculum;
- causal-relevance sampling;
- learned data-mixture optimization.

The proposed sample fields `quality`, `novelty`, `redundancy`, `uncertainty`, `difficulty`, and
`causal_relevance` are measured hypotheses, not trusted labels. Any weighted score is an exploratory
baseline. Keep a fixed data/example budget, measure quality versus total examples and compute, use
walk-forward or regime-held-out evaluation where relevant, shrink noisy estimates, correct repeated
tests, and never tune mixture weights on final test regimes. This track opens only after at least one
representation/dynamics/memory system is stable enough that changing its data has a clear
interpretation.

## 9. Data ladder

### R0: CMWM-Synth-E1

The first dataset is procedural and generated from versioned manifests. Its sealed specification
requires exact states, partial observations, rendered frames, event labels, counterfactual causal
annotations, and held-out mechanism/composition families. The current engineering scaffold has only
the symbolic portion and must pass G0 before it becomes an E1 dataset. CMWM-Synth-E1 is the only
dataset used for the E1 claim.

### R1: rendered CMWM-Synth-E1

Use the exact same manifests and queries with pixels instead of oracle state. This controls the
world and memory question while changing only the perceptual interface.

### R2: synthetic extensions

Add viewpoint/lighting shifts, irregular observation times, structural analogies, novel categories,
interactive agency, paired descriptions, multiple scripts, and synthetic glyph systems. Each
extension belongs to one pre-registered track; they are not introduced together.

### R3: bounded real-data transfer

Candidate families include natural images/video, Ego4D-style egocentric video, DROID robot
trajectories, and Open X-Embodiment. After controlled visual transfer, the Chen-lineage
[Sherlock](https://openneuro.org/datasets/ds001132) and
[Film Festival](https://openneuro.org/datasets/ds004042) naturalistic datasets may support an
external boundary/representation validation; they cannot validate CMWM's counterfactual causal
labels or replace controlled E1 testing. Selection, licenses, task labels, privacy, leakage, and
compute must be audited after the relevant synthetic gate passes. The earlier note named “Vistra”
without a resolvable dataset specification; it is not an approved dataset until it is identified
precisely.

### R4: scale

Larger video or multimodal corpora and 100M+ integration require successful controlled transfer.
No billion-parameter or foundation-scale run is authorized by synthetic success alone.

## 10. Ordered roadmap and hard gates

| Stage | One changed question | Output required to continue |
| --- | --- | --- |
| **E1 / R0** | Does oracle/learned event segmentation improve equal-budget long-delay memory from correct symbolic state? | G0, G1, and G2 pass |
| **P1B / R1** | Does that same effect survive frozen DINOv3 features from rendered frames? | G3 passes |
| **E2** | With the successful E1 writer fixed, does boundary-triggered reading beat equal-read periodic and random schedules? | New pre-registered read-timing gate; may open after G2 and the E1 report |
| **R2** | Which visual representation—CNN, ViT, frozen DINOv3, or in-domain DINO-style—best preserves controllable world state? | Pre-registered representation probe and efficiency gate |
| **R3** | Do CLIP-style alignment, shared/private latents, visual text, I-JEPA representation learning, or V-JEPA-style dynamics add value individually? | One decision/protocol and one passed gate per component |
| **R4** | Does learned structural graph memory beat vector retrieval on causal analogies? | Positive paired structural-retrieval effect with no oracle leakage |
| **R5** | Do fast weights/TTT/LoRA improve 1/2/5/10-shot adaptation without unacceptable forgetting or compute? | Adaptation and retention gate |
| **R6** | Can an alternative/local update rule reach equal quality more efficiently than BP? | Quality-to-total-resource gate |
| **R7** | Do consolidation, audio, data selection/mixtures, real trajectories, and integrated components retain their isolated gains? | Integration and interference gates before scale |

No stage is entered merely because its code exists. A failed hard gate stops or revises that branch;
it does not trigger a more complicated rescue architecture.

After every three to five completed experiments, run a meta-audit of task shortcuts, distribution
coverage, metric validity, compute accounting, and whether the synthetic environment still tests the
claimed capacity. A benchmark that has become saturated or gameable is revised under a new
protocol; its old test set is not used to tune the replacement.

## 11. Current work and authorization boundary

Proceeding now means:

1. finish G0 generator-validity, leakage, balance, and shortcut audits;
2. repair the current scaffold until G0 can certify diverse manifests and matched, non-trivial
   causal-recall candidates;
3. implement the conventional symbolic state adapter, GRU, query head, and equal-budget E1
   conditions;
4. prepare cluster environment instructions and inert Slurm configurations;
5. wait for explicit user authorization, login/environment information, storage paths, allocation,
   partition, QoS, and GPU constraints before any cluster connection, transfer, installation, or
   submission;
6. run no local experiment or model smoke test.

Code preparation is not cluster authorization. No `ssh`, `scp`, `rsync`, remote write, package
installation, or `sbatch` action is implied by this roadmap.
