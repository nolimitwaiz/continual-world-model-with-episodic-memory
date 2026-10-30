# Cluster execution

Training is fail-closed: it is permitted only inside a Slurm allocation with a visible CUDA
device. Local commands may validate configurations and inspect submission commands, but they must
not execute training.

Set machine-specific locations and scheduler choices through the environment or submit arguments:

```bash
export CMWM_DATA_ROOT=/cluster/path/to/cmwm-data
export CMWM_RUN_ROOT=/cluster/path/to/cmwm-runs
export CMWM_MODEL_CACHE=/cluster/path/to/model-cache
export CMWM_SLURM_ACCOUNT=my-account
export CMWM_SLURM_PARTITION=my-gpu-partition
# Optional when the target partition requires one:
export CMWM_SLURM_QOS=my-gpu-qos
```

Inspect a submission without creating files or invoking Slurm:

```bash
python scripts/submit_slurm.py \
  --config configs/prototype1/smoke.yaml \
  --dry-run
```

Inspect the ordered engineering smoke suite without invoking Slurm:

```bash
python scripts/submit_smokes.py --dry-run
```

The smoke submitter runs `environment`, `generator`, `g0-readiness`, and `components` serially with
`afterok` dependencies. `g0-readiness` is expected to block the component smoke until the dataset
generator and its audit sidecars satisfy G0; that block prevents premature model execution.

After every smoke passes, `slurm/attest_g0.sbatch` generates and audits the full declared dataset
and emits one sealed G0 artifact. Training must receive that exact artifact through
`CMWM_G0_ATTESTATION`; a smoke artifact cannot unlock training.

`slurm/audit_chen_metadata.sbatch` is a separate CPU-only metadata task for the active Chen Lab
scope. It runs fixture tests, queries immutable OpenNeuro snapshot descriptions/file trees, and
records declared counts and bytes. It never requests imaging or stimulus file contents and cannot
unlock model training.

Submit one configuration:

```bash
export CMWM_SUBMISSION_AUTHORIZED=explicit-user-authorization
python scripts/submit_slurm.py \
  --config configs/prototype1/context_only.yaml \
  --time 01:00:00 \
  --gpus 1 \
  --confirm-submit
```

The authorization token and `--confirm-submit` flag are both mandatory. Set them only after the
user has explicitly approved the target host, transfer, environment, allocation, storage paths, and
submission. Their presence is a software interlock, not a substitute for obtaining that approval.

The first trainer validation uses `configs/engineering/c0_context.yaml`, followed only after it
passes by the remaining entries in `configs/engineering/engineering_matrix.txt`. These configs
retain the exact G0-attested 10,000/1,000/1,000 episode contract while limiting optimization to 20
steps. They evaluate and atomically checkpoint every five steps. Every checkpoint stores model and
optimizer state, sampler and Torch/CUDA RNG state, code/config/G0 identities, elapsed time, peak
CUDA memory, and the metric history needed for exact resume.

After a training job is submitted, the operator reports the Slurm job ID and a conservative
check-back time, then ends the interactive research workflow task. Continuous polling is not the default. A
later user check-in triggers a fresh scheduler/result inspection.

For a job array, pass `--manifest` with one configuration path per non-empty line. Relative paths
in the manifest are resolved relative to the manifest file. The submitter validates and resolves
every configuration before invoking `sbatch`.

Account, partition, data paths, and run paths are deliberately absent from committed files. The
submission utility snapshots resolved configurations below `$CMWM_RUN_ROOT/_submissions` and
stores scheduler logs below `$CMWM_RUN_ROOT/slurm`.

For JHU ARCH, verify the account, partition, QoS, CPU-per-GPU ratio, and CUDA module on the target
cluster immediately before submission. The project does not assume that CLSP, Rockfish, and DSAI
share allocation names. Current ARCH guidance is linked from the project README.
