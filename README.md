# WeightWhatPP

Weight- and coefficient-based truncation rules for Pauli-operator propagation in
Trotterized quantum simulation — the code release of the PDS_PPML study.

We propagate the tilted transverse-field Ising model in the Heisenberg picture with
[PauliPropagation.jl](https://github.com/MSRudolph/PauliPropagation.jl), apply
layer-level truncation rules (coefficient threshold and weight caps), and cross-validate
the resulting `⟨Z_c(t)⟩` trajectory against an exact [Yao.jl](https://github.com/QuantumBFS/Yao.jl)
statevector simulation.

## Layout

- `src/` — the propagation pipeline (Julia). `pipeline.jl` includes the modules in order:
  `systems`, `truncation`, `io`, `trotter`, `metrics`, `propagation`, `reference`,
  `experiments`, `distributions`.
- `data/` — the lightweight JSON data extracted from the PauliSum snapshots,
  organized by system and truncation rule.
- `runs/` — local cache for raw `.jls` snapshots and extracted JSON files.  This
  folder is ignored by git.
- `scripts/` — small entry points for running propagation, extraction, and CDF
  export for one work item.
- `tools/` — utility scripts, including the builder for the public `data/` tree.
- `Project.toml`, `Manifest.toml` — the pinned environment (PauliPropagation 0.7.2,
  same git-tree as the production runs).

## Usage

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'include("src/pipeline.jl")'
```

## Data Pipeline

The full pipeline has two storage levels.

The raw level is `runs/`.  It contains serialized PauliSum snapshots:

```text
runs/<system>/<system>/<rule_hash>/snapshot_####.jls
runs/<system>/<system>/<rule_hash>/truncated_####.jls
```

These files are useful for re-extraction, but they can be huge, so they are not
committed.  The public level is `data/`.  It contains only the extracted JSON
files:

```text
data/<system>/<rule_hash>/config.json
data/<system>/<rule_hash>/summary.json
data/<system>/<rule_hash>/cdf_layers.json
```

Run one propagation task:

```bash
ITEM=0 EXPERIMENT=rect4x4 TASK=propagate julia --project=. scripts/run_one.jl
```

Extract metrics from the snapshots:

```bash
ITEM=0 EXPERIMENT=rect4x4 TASK=extract julia --project=. scripts/run_one.jl
```

Optionally extract coefficient and weight distributions:

```bash
ITEM=0 EXPERIMENT=rect4x4 TASK=cdf TARGET_ELLS=30,50,80,100 julia --project=. scripts/run_one.jl
```

Rebuild the committed `data/` folder from `runs/`:

```bash
tools/build_public_data.py --clean
```

For the production study, these same steps were run as arrays on SCITAS.  The
raw snapshots stay external; the code and extracted JSON data are kept here.
