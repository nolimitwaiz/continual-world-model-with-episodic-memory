> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# Cluster Execution Report — 2026-08-16

## Outcome

No model training was run. Two validation tasks completed on CLSP:

1. the full CMWM-Synth-E1 G0 dataset attestation; and
2. a metadata-only OpenNeuro audit for the active Chen Lab scope.

## Full synthetic G0 — job 1753878

- Slurm state: `COMPLETED`, exit `0:0`, elapsed `00:23:02`.
- Dataset: 12,000 episodes (10,000 train, 1,000 validation, 1,000 test).
- Queries: 97,548.
- G0 result: `passed: true`; zero errors.
- Finding: one warning, `collision_mechanics_not_evidenced`. Collision is
  outside the supported claim.
- Storage pressure: all 12,000 episodes caused eviction; all 18,281 primary
  query references were outside local context and had been evicted from the
  frame store.
- Random/oracle segment distributions: exact match reported by the audit.
- Manifest SHA-256:
  `b787ad5b1f1097de552ceb266a33506a4b8b95f6738d2eeb6fbf05b9ccf4e74c`.
- Seal SHA-256:
  `c497c58a1d86d1cd80ecf866e0826e800e31a8336479bd8e4594196eadcd0ad6`.
- Dataset-source SHA-256:
  `0626b52e3e1cd92e3295e127ac217cce239138189b7eb4fd06b3e59be4e6e60a`.
- Returned artifact SHA-256:
  `0468162f42f1cc18a1242115f0cc26dd0b16f165e8b05603f7b12c72a028babc`.

Artifacts: `results/cluster/1753878/` (historical source location: `results/cluster/1753878/`).

G0 makes only the matching declared synthetic configuration eligible for the
next pilot gate. It is not a model result and does not authorize a claim about
human memory or Chen Lab data.

## Chen OpenNeuro metadata audit — jobs 1753885 and 1753886

Job 1753885 failed in one second because the selected PyTorch environment did
not contain `pytest`. It did not reach the network query. The corrected job used
the existing dedicated test environment.

Job 1753886:

- Slurm state: `COMPLETED`, exit `0:0`, elapsed `00:00:03`.
- Fixture tests: three passed.
- Snapshots audited: FilmFest, Sherlock, Twilight Zone, and Merlin + Sherlock
  Listening.
- Total declared bytes: 66,623,473,114.
- Content downloaded: `false`.
- Returned JSON SHA-256:
  `2846253e054a196bfa92be0c1d4bda1d541e39d185a5be37f7e1d129906505c6`.

Artifacts: `results/cluster/1753885/` (historical source location: `results/cluster/1753885/`) and
`results/cluster/1753886/` (historical source location: `results/cluster/1753886/`).

## Complete unit suite — job 1753888

- Slurm state: `COMPLETED`, exit `0:0`, elapsed `00:00:08`.
- Result: 61 tests passed in 6.60 seconds.
- JUnit SHA-256:
  `8c780150a659e924351d75fc4745aa19460e1dbb732293868b040ab946992bea`.
- Log SHA-256:
  `234600ccf9efbec2f7dd364ed286a392e479f0a435a6faed70a7eb0efc038f51`.

Artifacts: `results/cluster/1753888/` (historical source location: `results/cluster/1753888/`).

## Next scientific gate

The next real-data action remains FilmFest. A confirmatory causal-retrieval
benchmark still requires verified human directed causal edges and a frozen
question manifest. The existing predicted graph can support only a separately
registered exploratory protocol.
