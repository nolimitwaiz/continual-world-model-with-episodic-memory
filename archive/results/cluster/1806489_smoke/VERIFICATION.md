> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# Smoke verification — 2026-09-26

Remote `verify_smoke.py` completed successfully before E3 matrix submission. It verified:

- Slurm job 1806489 completed with exit 0:0 in 6m29s (scheduler checked separately).
- Four readable checkpoints at steps 5, 10, 15, 20; finite model and optimizer tensors.
- Finite logged losses, gradients, and evaluation metrics.
- Matching executable/config/dataset/G0/endpoint identities.
- Four 512-row prediction files: 2,048 rows checked for unique IDs, frozen ordering,
  task/lag metadata, correctness flags, prediction digests, and recomputed accuracy.
- The final named checkpoint and `last.pt` have identical model tensors.
- Final result and final checkpoint/learning-curve evaluation agree.
- The unchanged source passes the trainer's G0 and evaluation-manifest checks for all five configs.

Final smoke LDMS: 0.271484375. Measured trainer execution: 344.41239293292165 seconds.
This score is an engineering diagnostic only. No hyperparameter or seed selection used this result.

Validator source: `cluster/executions/e3_20260926/verify_smoke.py` (outside frozen executable tree).
Frozen executable SHA-256: `067f49cc162929d935b7c1b497eb3c8131d7a1662358ceb1360a11ff623b3cec`.
No test data were accessed; no training or experiments ran locally.
