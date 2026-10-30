> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# CMWM Publication and Experiment Roadmap

Status: active program registry, reconciled 2026-08-24.

This document turns the full idea set into paper-sized, falsifiable studies. “Run all experiments”
means retain all justified studies in this registry and execute them in dependency order. It does
not mean combine multiple unvalidated mechanisms in one model.

## Paper program

### Paper 1 — Event-structured episodic memory

Question: under matched finite storage and read budgets, does event-aligned grouping improve
long-delay state and causal recall over context, frames, fixed chunks, and length-matched random
chunks?

Required conditions: C0, C1, C2, C3, C4; C5 only after oracle value is demonstrated.

Required result package:

- five preregistered seeds;
- paired confidence interval for Long-Delay Memory Score;
- separate lag 16/32/64/128 and task-family results;
- positive direction on HF-A and HF-B;
- storage, retrieved-record, parameter, update, FLOP, wall-time, and peak-memory accounting;
- boundary-rate and segment-length controls;
- error analysis and shortcut meta-audit;
- complete configs, source/data hashes, checkpoints, predictions, and negative results.

Primary falsifier: C4 fails to beat C3 and the strongest simpler baseline under the locked gate.

### Paper 2 — Slow weights, fast episodic state, and fast weights

Entry gate: Paper 1 identifies a stable memory condition and a continual task sequence is frozen.

Factorial core:

| Episodic state | Fast weights | Meaning |
| --- | --- | --- |
| off | off | slow-weight/context baseline |
| on | off | event memory only |
| off | on | TTT/online adapter only |
| on | on | combined system |

Candidate fast mechanisms enter one at a time: online LoRA, TTT inner updates, a small fast-weight
layer, then a learned update rule. Standard backpropagation remains the outer-loop control.

Endpoints: 1/2/5/10-example adaptation, forward transfer, backward transfer, retention, forgetting,
calibration, update FLOPs, wall time, memory, and recovery after distribution shifts.

Consolidation is a second factorial after the core: no replay, uniform replay, event-boundary replay,
and importance/novelty-aware replay into slow weights. A combined system is retained only if gains
survive interference and resource accounting.

### Paper 3 — Multimodal transfer and real-world validation

Entry gate: the symbolic memory effect and learned online boundary mechanism pass.

First transfer instrument: frozen DINOv3 dense features on rendered versions of the same sealed
episodes. It tests survival of the memory effect under perceptual noise, not whether DINO is best.

Later isolated representation studies:

- small CNN versus small ViT under matched parameters/compute;
- frozen DINOv3 versus frozen CLIP image tower with identical adapters, while disclosing unmatched
  upstream pretraining;
- in-domain DINO-style self-supervision versus the same backbone trained conventionally;
- I-JEPA-style static-image objective;
- per-frame DINO plus GRU/Transformer versus V-JEPA-style video features;
- action-conditioned V-JEPA-style prediction only on a frozen interactive planning task.

CLIP-style image/text alignment, shared/private latents, symbolic text, pixel orthography, audio,
and real trajectories each receive a separate baseline and entry gate. None is added merely to make
the architecture look multimodal.

### Paper 4 candidate — Human event memory and neural correspondence

Active Chen-family scope: FilmFest, Sherlock, The Twilight Zone, Pieman, Tunnel Under the World,
Green Eyes, and Merlin + Sherlock Listening.

Behavioral order: FilmFest causal/event-network benchmark, then Sherlock graph-free external
validation, then separately frozen generalization questions for the remaining releases.

Neural work starts only for a named preregistered hypothesis, for example boundary-linked
reactivation, event representation correspondence, or subject-held-out encoding. fMRI is not used
to claim patient-style word decoding without the temporal resolution, targets, sample size, and
subject-generalization evidence required for that distinct problem.

## Complete experiment registry

| ID | Single changed question | Mandatory control | Entry gate | Paper role |
| --- | --- | --- | --- | --- |
| E1 | Does event grouping improve finite episodic memory? | C0/C1/C2/C3 | G0 passed | Paper 1 core |
| E2 | Do boundary-triggered reads help independently of writes? | periodic/always-on/random matched equal reads | E1 writer retained | Paper 1 extension |
| E3 | Do boundary-biased encoding strengths help? | uniform write strength | E1 retained | Chen mechanism ablation |
| E4 | Does event-network centrality improve replay/eviction? | FIFO/vector, no oracle graph | learned graph available | structural memory |
| E5 | Does agency improve boundary/retrieval decisions? | identical passive trajectories | agency benchmark frozen | Chen extension |
| GI1 | Does invariant structural retrieval beat vector retrieval? | equal-byte/top-k vector retriever | E1 report + no leakage | graph paper/extension |
| T1 | GRU versus Transformer | same memory/data/interface | Paper 1 robustness | robustness |
| T2 | CfC under irregular time | GRU/Transformer with `delta_t` | asynchronous dataset | dynamics |
| T3 | fixed recursive depth | one pass, 2/4/8 tied iterations | depth-controlled tasks | reasoning |
| T4 | adaptive stopping | strongest fixed-depth recursion | T3 positive | reasoning efficiency |
| F1 | Fast weights versus episodic state | 2x2 factorial | continual sequence | Paper 2 core |
| F2 | TTT versus online LoRA versus small fast layer | same adapter budget | F1 need established | fast adaptation |
| F3 | Replay/consolidation into slow weights | no/uniform replay | F1 stable | continual learning |
| R1 | CNN versus ViT | matched data/params/compute | rendered dataset | representation |
| R2 | DINOv3 transfer | same frozen encoder for all memory conditions | C5 pass | Paper 3 bridge |
| R3 | DINO-style versus I-JEPA-style objective | same backbone/data/compute | R1 established | self-supervision |
| R4 | DINO image tower versus CLIP image tower | identical adapter/downstream | grounded task | representation |
| V1 | Per-frame DINO dynamics versus V-JEPA-style video | identical clips/actions/heads | visual state stable | video dynamics |
| M1 | Independent image/text encoders versus CLIP alignment | late fusion | paired task frozen | multimodality |
| M2 | Shared/private decomposition | single shared vector | M1 positive | multimodality |
| L1 | subword/character versus rendered pixel text | matched semantics | multilingual task frozen | language |
| A1 | frozen audio interface and alignment | text/image baseline | audio task frozen | multimodality |
| U1 | backprop versus alternative update rule | full BP | small stable benchmark | learning rules |
| U2 | learned updater and global-correction frequency | true BP teacher | U1 stable | credit assignment |
| D1 | uniform versus quality/novelty/redundancy sampling | fixed examples/compute | stable model | data efficiency |
| D2 | uncertainty/difficulty/causal curriculum | uniform + one factor | D1 stable | data intelligence |
| H1 | FilmFest behavioral causal retrieval | lexical/dense/long context | human edges frozen | human validation |
| H2 | Sherlock event/reinstatement transfer | graph-free baselines | H1 protocol complete | external validation |
| H3 | fMRI event correspondence | subject-held-out encoding controls | named neural hypothesis | neural validation |

Forward-Forward, feedback alignment, synthetic gradients, forward gradients,
predictive-coding-inspired updates, NoProp, and local/global hybrid updates remain in U1/U2. They
are not forgotten, but none enters before the backprop benchmark is stable.

## Execution discipline

1. One changed variable per comparison.
2. Strong conventional baselines precede exotic candidates.
3. Pilots choose feasibility and budgets; they never become confirmatory evidence retroactively.
4. Freeze the confirmatory protocol, seeds, manifests, metrics, and hardware class before test use.
5. Report null and negative results; do not rescue a failed mechanism by silently altering the test.
6. After every three to five experiments, audit shortcuts, coverage, metric validity, and compute.
7. Integrate only independently retained components in a later factorial interference study.
8. Do not use a larger model or dataset to compensate for a failed controlled mechanism.

## Resource plan

- CPU Slurm jobs: data/QC, unit tests, graph audits, preprocessing, statistics, checkpoint comparison.
- General GPUs with CUDA-12-compatible PyTorch: symbolic training and most small ablations.
- Parallelism is across conditions and seeds, one GPU per run—not data parallelism for the tiny E1
  model.
- A100s are reserved for a demonstrated resource requirement in visual/video or scale studies.
- Each artifact records node, GPU, PyTorch, CUDA, source, config, dataset seal, seed, timing, and peak
  allocation. Efficiency comparisons use only matched GPU classes.

## Current critical path

```text
general-GPU environment
  -> exact resume
  -> corrected C0-C4 pilot
  -> resource/profile freeze
  -> five-seed C0-C4 confirmation
  -> C5 learned boundaries if C4 wins
  -> Paper 1 result and decision report
  -> E2 / fast weights / DINO transfer as separately gated studies
```

