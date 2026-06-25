# WeightWhatPP

Which Pauli strings are worth keeping?

This repository contains the code and extracted data used to benchmark
truncation rules for Pauli operator propagation under Ising dynamics.  The
question is practical: if a Pauli expansion is growing too fast, which
truncation rule gives the best accuracy for the memory it spends?

The simulations use
[PauliPropagation.jl](https://github.com/MSRudolph/PauliPropagation.jl) in the
Heisenberg picture, with exact state-vector references from
[Yao.jl](https://github.com/QuantumBFS/Yao.jl) when the system is small enough.

The three truncation families are:

- coefficient-only truncation;
- coefficient truncation plus a total Pauli-weight cap;
- coefficient truncation plus an `XY`-weight cap.

The benchmark systems are periodic rings and open rectangular lattices for the
tilted transverse-field Ising model.

## What Is Here

```text
.
├── src/          Julia code for systems, circuits, propagation, truncation,
│                references, metrics, and extraction
├── scripts/      Small launchers for local runs or Slurm arrays
├── data/         Public JSON data extracted from the raw snapshots
├── tools/        Utilities for rebuilding the public data folder
├── runs/         Local raw snapshot cache; ignored by git
├── Project.toml  Julia environment
└── Manifest.toml Pinned dependency versions
```

Think of `runs/` as the heavy machinery and `data/` as the clean export.  The
raw `.jls` PauliSum snapshots can become huge, so they are not committed.  The
repository keeps the extracted JSON files needed for analysis and plotting.

## Quick Start

```bash
git clone https://github.com/IzHug/WeightWhatPP.git
cd WeightWhatPP
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Load the pipeline:

```bash
julia --project=. -e 'include("src/pipeline.jl")'
```

Run one work item:

```bash
EXPERIMENT=rect4x4 ITEM=0 TASK=propagate julia --project=. scripts/run_one.jl
```

The same launcher is used for propagation, extraction, and distribution data:

```bash
# Generate raw PauliSum snapshots.
EXPERIMENT=rect4x4 ITEM=0 TASK=propagate julia --project=. scripts/run_one.jl

# Extract scalar diagnostics from the snapshots.
EXPERIMENT=rect4x4 ITEM=0 TASK=extract julia --project=. scripts/run_one.jl

# Extract coefficient and weight distributions for selected layers.
EXPERIMENT=rect4x4 ITEM=0 TASK=cdf TARGET_ELLS=30,50,80,100 julia --project=. scripts/run_one.jl
```

Use `TASK=both` to propagate and extract in one pass.

## Experiments

The main experiment names are:

```text
chain16
rect4x4
rect4x4_h1_g0p5
rect5x5
rect5x5_h5p158
rect4x4_dt0p04_l25
rect5x5_dt0p04_l25
rect11x11_dt0p04_l25
```

`ITEM` is zero-based for local runs.  On a Slurm array, the script reads
`SLURM_ARRAY_TASK_ID` automatically.

## Data

Each system in `data/` has an `index.json` and one folder per truncation rule.
Rule folders are named by stable hashes.

```text
data/<system>/<rule_hash>/config.json
data/<system>/<rule_hash>/summary.json
data/<system>/<rule_hash>/cdf_layers.json
data/<system>/reference/yao_overlap.json
```

- `config.json` describes the physical system and truncation rule.
- `summary.json` stores scalar diagnostics, including sizes and scores.
- `cdf_layers.json` stores coefficient and weight distributions, when present.
- `reference/yao_overlap.json` stores the exact Yao reference, when available.
- `data/manifest.json` lists the systems in the public data release.

To rebuild the public data folder from a local `runs/` cache:

```bash
tools/build_public_data.py --clean
```

## Storage Model

The pipeline has two layers:

```text
runs/    raw snapshots, large, local only
data/    extracted JSON, small enough to publish
```

This is deliberate.  The snapshots are useful if you want to re-extract a new
diagnostic, but the JSON files are the stable public artifact used by the
report.

## Compute

The production runs were executed on the Jed cluster of EPFL SCITAS.  Standard
Jed CPU nodes have 72 cores, using two 36-core Intel Xeon Platinum 8360Y
processors.

## Status

This is a research code release, not a polished Julia package.  The pinned
environment is included so the benchmark pipeline can be reproduced as closely
as possible.
