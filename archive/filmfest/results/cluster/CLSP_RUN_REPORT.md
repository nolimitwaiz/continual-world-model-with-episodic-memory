> Historical research record. Describes the state at its original date, not the current release. See the research report for current evidence boundaries. Machine paths have been redacted.

# CLSP Cluster Run Report

Date: 2026-08-11  
Cluster account: `<CLUSTER_ACCOUNT>`  
Scheduler: Slurm 23.11.11  
Remote run root: `<CLUSTER_PATH>`

## Outcome

The locked FilmFest downstream reproduction completed successfully on CLSP. The cluster produced a byte-identical normalized CSV and reproduced the macOS numerical results to a maximum absolute floating-point difference of `6.22e-15`. All 10 tests and all six publication comparisons passed.

This establishes cross-platform reproducibility on macOS ARM64 and Linux x86-64. It does not remove the missing-input blockers for upstream semantic and causal graph reconstruction.

## Slurm runs

| Job | Purpose | Resources | State | Elapsed |
|---|---|---|---|---:|
| 1752454 | A100 device inventory | `gpu-a100`, 1 GPU, 4 CPUs, 8 GB | COMPLETED | 1 s |
| 1752455 | First FilmFest attempt | `cpu`, 2 CPUs, 4 GB | FAILED before experiment | 3 s |
| 1752460 | FilmFest reproduction retry | `cpu`, 2 CPUs, 4 GB | COMPLETED | 23 s |
| 1752497–1752501 | Representative hardware inventory | five one-GPU, one-CPU probes | COMPLETED | <1 s each |
| 1752503 | Unavailable e05-family probe | one requested GPU | CANCELLED while pending | 0 s |

Job 1752455 failed because the compute-node system Python lacked `ensurepip`. No scientific output was created. The recorded fix uses the user's existing `uv` installation and an isolated Python 3.12.13 environment. First-time dependency resolution/download/installation took approximately 12 seconds; the complete successful job took 23 seconds.

## Accessible resource inventory

The account has these relevant Slurm associations:

- `default` account for the general `cpu` and `gpu` partitions;
- `a100acct` for the dedicated `gpu-a100` partition.

The A100 partition exposes 28 generic GPU resources across four nodes: three 96-CPU/approximately 1 TiB nodes with eight GPUs each and one 256-CPU/approximately 2 TiB node with four GPUs. The latter node was unavailable during the probe.

Exact sampled hardware:

| Accessible family | Sample node | CPU | System RAM | GPU |
|---|---|---|---:|---|
| Older general GPU | c04 | Intel Xeon E5-2620 v3, 24 logical CPUs | 62 GiB | GTX 1080 Ti, 11 GiB |
| General GPU | c14 | Intel Xeon Silver 4110, 32 logical CPUs | 62 GiB | GTX 1080 Ti, 11 GiB |
| Larger-memory general GPU | c24 | Intel Xeon Silver 4110, 32 logical CPUs | 125 GiB | RTX 2080 Ti, 11 GiB |
| Eight-GPU general node | octopod | Intel Xeon Silver 4210R, 40 logical CPUs | 125 GiB | GTX 1080 Ti, 11 GiB |
| A100 family | e03 | AMD EPYC 7443, 96 logical CPUs | approximately 1 TiB | NVIDIA A100-SXM4-80GB, 80 GiB |

The A100 diagnostic ran on e03 with driver `580.159.03`. Slurm reports e02–e04 as the eight-GPU A100 family and e05 as the four-GPU/2-TiB family. CPU/GPU hardware in restricted `reserve_q`, `iarpa_arts`, and `jsalt26` partitions is not counted as available to this user because the audited Slurm associations do not grant those accounts.

## Scientific results

| Check | CLSP result | Local result | Difference |
|---|---:|---:|---:|
| Semantic centrality–recall r | 0.20242360841142457 | 0.20242360841142445 | 1.1e-16 |
| Causal centrality–recall r | 0.29409889375176107 | 0.29409889375176070 | 3.9e-16 |
| Semantic high–low t | 6.119080700782461 | 6.119080700782461 | 0 |
| Causal high–low t | 8.229110922270156 | 8.229110922270156 | 0 |
| Semantic Cohen's dz | 1.579939843215312 | 1.579939843215312 | 0 |
| Causal Cohen's dz | 2.1247473037363913 | 2.1247473037363913 | 0 |

The normalized CSV is byte-identical across machines. Every result file other than the checksum manifest itself passed SHA-256 verification on the original cluster paths. The future Slurm script excludes the manifest from hashing itself.

## Resource decision

The successful scientific job correctly used CPUs. The current scripts perform workbook I/O, small dataframe operations, correlations, and SVG generation; they contain no CUDA kernels. An A100 allocation would not accelerate them.

Use A100s for:

1. a verified exact USE environment if the pinned artifact can use modern CUDA, otherwise use a CPU/container for legacy compatibility;
2. Song predecessor training/reproduction;
3. future multi-seed dense-retriever or neural baseline training;
4. later authorized neural phases.

Do not request multiple GPUs until a workload implements distributed data/model parallelism and a one-GPU scaling baseline exists. More GPUs do not automatically make a serial program faster.

## Local artifacts

- `1752454/a100-inventory-1752454.out` (historical source location: `./1752454/a100-inventory-1752454.out`)
- `1752460/filmfest-first-attempt-1752455.out` (historical source location: `./1752460/filmfest-first-attempt-1752455.out`)
- `1752460/filmfest-1752460.out` (historical source location: `./1752460/filmfest-1752460.out`)
- `1752460/artifacts/reproduction_evaluation.json` (historical source location: `./1752460/artifacts/reproduction_evaluation.json`)
- `1752460/artifacts/tests.stderr.txt` (historical source location: `./1752460/artifacts/tests.stderr.txt`)
- `hardware/` (historical source location: `./hardware/`)
