# Extracted data

This folder contains the lightweight data extracted from the PauliSum
snapshots.

The raw snapshots are not stored in GitHub.  They are Julia `.jls` files written
by the propagation step, and the full production snapshot tree is too large for
this repository.  The files kept here are the JSON outputs needed to reproduce
the analysis and figures.

## Layout

```text
data/
├── manifest.json
├── chain16/
│   ├── index.json
│   ├── reference/
│   │   └── yao_overlap.json
│   └── <rule_hash>/
│       ├── config.json
│       ├── summary.json
│       └── cdf_layers.json
├── rect4x4/
└── ...
```

Each system has one folder.  Each truncation rule has one folder named by its
stable rule hash.  The readable rule name, the physical system, and the
truncation parameters are stored in `config.json` and summarized in
`index.json`.

## File Meaning

- `config.json`: system parameters and truncation-rule parameters.
- `summary.json`: scalar diagnostics extracted from the PauliSum snapshots.
- `cdf_layers.json`: coefficient and weight distributions for selected layers,
  when available.
- `reference/yao_overlap.json`: exact Yao reference data, when available.
- `manifest.json`: top-level index of the systems included in `data/`.

## From Snapshots To Data

The pipeline is:

1. Generate raw PauliSum snapshots with the code in `src/`.

   ```bash
   ITEM=0 EXPERIMENT=rect4x4 TASK=propagate julia --project=. scripts/run_one.jl
   ```

   This writes `snapshot_####.jls` and `truncated_####.jls` under `runs/`.

2. Extract JSON diagnostics from those snapshots.

   ```bash
   ITEM=0 EXPERIMENT=rect4x4 TASK=extract julia --project=. scripts/run_one.jl
   ```

   This writes `summary.json` in the same run directory.

3. Optionally extract distribution data for selected layers.

   ```bash
   ITEM=0 EXPERIMENT=rect4x4 TASK=cdf TARGET_ELLS=30,50,80,100 julia --project=. scripts/run_one.jl
   ```

   This writes `cdf_layers.json` when the required snapshots are present.

4. Rebuild the public `data/` tree from the local `runs/` cache.

   ```bash
   tools/build_public_data.py --clean
   ```

For the full production study, the same steps were run as Slurm arrays on
SCITAS.  The generated snapshots stay outside the repository, while the
extracted JSON files are committed here.
