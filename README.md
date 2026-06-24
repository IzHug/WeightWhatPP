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
- `Project.toml`, `Manifest.toml` — the pinned environment (PauliPropagation 0.7.2,
  same git-tree as the production runs).

## Usage

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'include("src/pipeline.jl")'
```
