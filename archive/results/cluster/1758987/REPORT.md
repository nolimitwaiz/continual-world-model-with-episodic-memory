> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# C0 checkpointed trainer attempt 1 — failed before step 1

Date: 2026-08-22  
Slurm job: 1758987  
Classification: engineering; not claim-eligible

## Outcome

The job reached an NVIDIA A100-SXM4-80GB and loaded the exact G0-attested
configuration, but failed during the first context-only forward pass before an
optimization step or checkpoint. Autocast produced a half-precision empty
retrieval-score tensor, and padding it with the hard-coded value `-1e9`
overflowed that dtype.

```text
RuntimeError: value cannot be converted to type c10::Half without overflow
```

No model-performance inference is permitted. No checkpoint, metric history, or
result artifact was produced. The status file remained `running`, which also
exposed missing runtime-exception status handling.

## Repair boundary

The retry changes only numerical masking and failure recording:

- every masking/padding sentinel is the finite minimum representable by the
  active tensor dtype;
- runtime exceptions atomically write a terminal `failed` status;
- float16, bfloat16, and float32 sentinel regression fixtures are included.

The data, model, condition, seed, optimizer, batch size, steps, checkpoint
cadence, and metric remain unchanged. The original log and status are archived
beside this report; the retry uses a separate dated cluster run root.
