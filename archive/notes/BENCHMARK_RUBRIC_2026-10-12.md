> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# CMWM small-validation benchmark and assessment rubric

Status: DRAFT, 2026-10-12. Design and local engineering validation only. No training, dataset-scale
generation, model evaluation, paid API call, upload, cluster connection or job submission performed.
This is our diagnostic benchmark proposal, not an established public benchmark or a frozen G1 test.

## 1. Where we actually stand

Latest verified research result remains E3 (September 28): 25 final runs, event grouping 73.6%,
fixed 56.3%, random-matched 55.7% on the same development endpoint. Boundaries were supplied;
gains were task-dependent and four of five seeds beat the chunk baselines. Earlier E2 was null.
Sherlock's separate BM25 v2 development primary was null. Raw fMRI was not validated.
September 30 added setup/scoring primitives and 17 small engineering tests, not new experiments.
G1 confirmation is incomplete; the legacy held-out generator path needs repair. The full world
model, learned segmentation, raw speech/vision pathway and true fast weights are not validated.

This request changes near-term priority to small, rubric-driven speech/vision validation. Preserve
all old results and the master plan. Pause expansion of the Sherlock text-embedding follow-up;
do not delete it. The larger 25-checkpoint/1,024-question confirmation plan is not the next launch.
No inference permission is implied by a draft protocol or by passing local code tests.

## 2. Two hypotheses, not one uncontrolled architecture change

H-memory: Under matched visible information, record bytes and reads, event-organized memory
improves delayed-state, hidden-state, temporal-order and simple next-state accuracy over fixed and
length-matched random records. Keep the perception interface and temporal core fixed within this
comparison. Supplied boundaries remain privileged information and are labeled oracle conditions.

H-representation: A non-Transformer visual/acoustic pathway can preserve the task-relevant
information at an acceptable cost. Test this FIRST on immediate perception without long delays.
Poor perception cannot diagnose failure of episodic memory. Changing an encoder and memory writer
together cannot isolate either hypothesis. Successful small validation is not proof of G1/G3.

Provisional interpretation of the user: ordinary visual frames and acoustic speech, not rendered
text images. A clarification is pending. No OCR, transcripts, captions or text embeddings feed our
observation path. Evaluation labels and task IDs are metadata, not model-visible answers.
For a strictly no-text language interface, queries must also be spoken; until a query decoder is
ready, fixed task selectors probe capabilities without claiming open-ended language understanding.

## 3. Architecture candidates, not implementations or results

First proposed baseline: visual CNN + recurrent GRU state + existing bounded event memory.
Second, separately: waveform 1-D CNN or log-mel CNN + GRU + the same memory interface.
Then paired audiovisual input, only after the unimodal probes work. Log-mel is an acoustic
representation, not transcription. CNNs/GRUs still encode information: the meaningful distinction
is no text-token input pathway and no Transformer backbone, not no encoding of any kind.

A selective state-space model is a later alternative temporal core, not an extra simultaneous
component. DINO/CLIP/Whisper or other attention-based backbones cannot be smuggled into a strictly
non-Transformer condition. Upstream pretrained architecture and supervision must be disclosed.
An untrained CNN+GRU is NOT a serious GPT baseline; a capable checkpoint or separately authorized
small train/validation-separated adapter experiment is required. No such checkpoint is asserted
available. Pixel-language representations are a different hypothesis from ordinary video and are
deferred pending clarification; using pixels by itself does not remove Transformer architecture.

## 4. Benchmark packet: small by construction

Name: CMWM-MiniAV-v0 (internal diagnostic).

Run one modality at a time: vision first, speech second, audiovisual last. Each proposed packet
has eight setup episodes and 32 scored validation episodes, with six question types per episode
(48 setup / 192 scored questions). These are caps and draft design counts, not existing data.
Setup clips are never counted in scored validation. Reusing a development packet for tuning
requires a fresh later confirmation set; it cannot become confirmatory retrospectively.

Use short two-object/two-container event sequences with confusable appearances and randomized
bindings, matched distractors, post-reference changes, and balanced candidate locations. At least
half the memory questions must require information absent from the final observation. Interleave
short/long physical delays (draft 2 and 8 seconds, exactly 16 scored episodes each); report actual
elapsed time and number of observations, not just nominal frames. No frame-rate-dependent proxy.
Speech uses spoken event descriptions and controlled queries; a written authoring script is
evaluator-only. Voices/phrasing/backgrounds must not reveal answers. Do not use environmental sound
recognition as a substitute for demonstrating speech understanding.

Sources: initially audited controlled clips for exact truth and new-scene generation. A later small
CLEVRER slice offers an external *synthetic* visual-reasoning check, not real-world validation.
Chen-family naturalistic movie/audio remains the intended real-data bridge after stimulus access,
rights, timing and task-label review. Do not claim raw media are ready just because behavioral
annotations exist. No bulk fMRI acquisition or public-media upload is authorized.

Answers: four answer options per question, including an 'insufficient information' option where
appropriate. Balance correct option positions within task and delay; compute chance separately
for each task. The dedicated unobservable task has abstention as its correct semantic answer,
so report balanced abstention discrimination against the matched answerable questions as well.
Option positions and labels are never used as physical object identities.

Future-state questions reveal only an observation prefix and, where needed, the same next-action
cue to every system. If the next state is not determined by what was observed, the correct answer
is uncertainty, not privileged simulator truth. This is a tiny predictive probe, not generative
video modeling or proof of causal reasoning.

## 5. Objective scorecard: do not hide failure in one weighted score

| Dimension | Example | Objective score / diagnostic |
|---|---|---|
| Immediate perception | Which box currently contains the visible ball? | Exact answer; prerequisite diagnostic |
| Delayed state | Where was it before the move? | Exact answer at each delay |
| Hidden-state continuity | Where should it be after the box is covered? | Exact answer, only when observable history supports it |
| Temporal order | Did moving happen before closing? | Exact event-order answer |
| Next-state prediction | Given the shown action, where will it be next? | Exact structured next-state choice; uncertainty when underdetermined |
| Unobservability | It moved off camera; which box now? | Correct abstention plus false-abstention rate on answerable items |
| Grounding | Which earlier moment supports the answer? | Evidence-span overlap diagnostic plus human support check |
| Robustness | Same event, changed appearance/voice/background | Paired accuracy drop; count variants as paired, not new independent episodes |
| Efficiency | What did remembering cost? | Stored bytes, reads, input frames/audio seconds, p50/p95 latency, peak memory, API cost |

Primary *pilot* metric: equal-weight mean of delayed state, hidden-state continuity, temporal
order and next-state exact accuracy, with every component and delay shown separately. Perception
and unobservability remain prerequisites/diagnostics, not extra easy items that inflate the score.
Wrong, missing, malformed and timed-out answers score zero; missing answers do not leave the
denominator. Record failures separately. Exact labels are scored deterministically, not by a GPT
judge. For probabilities, collect a full distribution over the actual choices and use a proper
multiclass score; do not label an arbitrary confidence number a calibrated probability.

Primary memory contrast: event versus length-matched random, with fixed-chunk comparison mandatory.
Do not mix this new pilot metric with old E3 LDMS or relax any pre-existing G1 threshold.

## 6. Human rubric: secondary diagnosis, not a substitute primary endpoint

Two reviewers, blind to model/condition and each other's scores, review the same fixed 24-case
subset (four per question type) after reading anchored examples. Selection is frozen before model
outputs and includes all declared input types. Report agreement, raw disagreements and adjudicated
scores separately; do not rewrite the primary answer key after seeing a winning model's output.

| Rubric dimension | 0 | 1 | 2 |
|---|---|---|---|
| Factual fidelity | Wrong or contradicts observed facts | Incomplete/mixed, if a free response was requested | Correct, with no invented detail |
| Temporal fidelity | Wrong event/order | Relevant events but ambiguous order | Correct past/present/next relation |
| Evidence support | Absent, fabricated or contradictory evidence | Relevant moment, insufficient support | Observed evidence directly supports the answer |
| Uncertainty | Invents an answer to an unknowable question | Expresses doubt but still asserts unsupported details | Appropriately abstains, or answers when evidence suffices |

Use N/A for rubric dimensions not requested or inapplicable; do not assign an automatic 2.
No eloquence, verbosity or human-likeness points. A polished explanation cannot rescue a wrong
structured answer. Do not request or grade hidden chain-of-thought.

## 7. Baselines and GPT: two different comparison tables

Controlled memory table: same usable perception checkpoint, temporal model, data, context,
training schedule and hardware; compare no episodic memory, fixed chunks, random length-matched
chunks and oracle events. Preserve equal byte/read budgets among memory-bearing conditions.
Log retained temporal coverage; equal record counts alone do not ensure equal historical access.
Context-only and memory-bearing controls differ intentionally in storage; report that distinction.

External GPT table: one image-capable GPT on vision items; one audio-capable GPT on speech items.
Candidate IDs checked in official docs on 2026-10-12: `gpt-6-luna` (text/image input) and
`gpt-audio-1.5` (text/audio input). Neither candidate supplies a verified joint image+audio interface
here. Do not send audio to the vision model or convert it to text silently. Models, API behavior,
availability, credential choice, cost cap and returned model version need approval/checking before
implementation; no API client has been authored or invoked in this phase.

If 'GPT-1' means the original historical model, it is not a modality-matched main baseline for
vision/speech. Do not silently reinterpret a request for a particular GPT version; confirm first.

For each supported modality compare (a) a full available history reference and (b) a matched
visible-history reference. Same frames/timestamps/audio segment/prefix, same task/candidates, no
future observations, tools or browsing. A fixed textual task instruction for the external GPT is
interface scaffolding, not a caption/transcript of observations; disclose it. A strictly audio-only
query regime needs a compatible interface for every comparator. Record all serialization and
sampling differences, and verify that the chosen frame schedule still exposes the evidence.

Full-history GPT is a privileged-information reference, not a guaranteed ceiling. GPT pretraining,
parameter count, internal compute and memory cannot be matched to a small research model. Therefore
no same-compute, architecture-superiority, or 'better than GPT/RAG generally' conclusion follows.
If snapshots are not immutable/dated, log returned IDs, date, request settings and this limitation.
One frozen prompt per task, capped output, and zero validation-driven prompt search. Repeat a
prespecified eight-case subset only for decoding variability; repeats are not independent data.

## 8. Rubric decisions and stop rules

1. DATA: all cases receive a gold-answer/observability/evidence audit; missing evidence, split
   overlap, leaked answers, duplicate scenes across splits, or changed input hashes blocks scoring.
2. INTERFACE: all eight setup episodes must have correct modality, timing, prefix and option
   mapping; malformed outputs stay failures, not automatic retries until they succeed.
3. PERCEPTION: draft engineering target >=90% immediate-perception accuracy and no sensory
   stratum below75%. If it fails, diagnose representation before interpreting memory scores.
   These pragmatic targets are NOT scientific significance thresholds and require pre-run freeze.
4. PILOT: report episode-paired uncertainty and task/delay breakdowns. Three paired fitted seeds
   are proposed only if a capable trainable model path exists; no training is currently authorized.
   Otherwise label a single-checkpoint run a feasibility diagnostic, not replicated evidence.
5. NEXT STEP: a mean oracle-minus-random gain >=5 percentage points with positive direction in
   each of three fitted seeds is only a proposed reason to design a larger confirmation. Require
   the fixed-chunk contrast and all validity/resource checks; no automatic scale-up. Wide intervals
   mean inconclusive. A small pilot cannot pass G1 or establish clinical/neural/real-world claims.
6. STOP: no selective seed replacement, metric switching, larger models or new components to
   rescue a result. Diagnose failed perception separately from failed memory.

Immediate cap: one modality, one architecture, four grouping controls, eight setup +32 scored
episodes. GPT external calls have a zero-dollar authorized budget for now. Any small adapter
training, dataset generation or inference run still requires explicit cluster submission approval.

## 9. What to do next, before spending compute

Freeze this scorecard and confirm the pixel-input interpretation. Hand-specify the eight setup
stories with truth, timing, answerability and answer options. Build/inspect the observation packet
and identify a viable non-Transformer perception checkpoint/adapter. Only then request a small
cluster validation budget and, separately, an API/model/credential/cost decision. No full-stack
model build, multimodal fusion sweep, long training run or expansion of text encoding is next.

## Sources checked

- OpenAI evaluation design: https://developers.openai.com/api/docs/guides/evaluation-best-practices
- Image reference candidate: https://developers.openai.com/api/docs/models/gpt-6-luna
- Audio reference candidate: https://developers.openai.com/api/docs/models/gpt-audio-1.5
- CLEVRER diagnostic video benchmark: https://arxiv.org/abs/1910.01442
- Mamba selective state-space alternative: https://arxiv.org/abs/2312.00752
- Pixel-language counterexample (uses Transformers): https://huggingface.co/Team-PIXEL/pixel-base
