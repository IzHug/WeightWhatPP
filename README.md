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

## Conventions

The study Hamiltonian carries negative signs,

```
H = -J Σ⟨i,j⟩ Z_i Z_j  -  h Σ_i X_i  -  g Σ_i Z_i ,
```

so the Schrödinger Trotter gate is `exp(-i H_term Δt) = exp(+i coupling Δt P)` and, with
`PauliRotation(θ) = exp(-iθ/2 P)`, the rotation angles are `θ = -2·{J,h,g}·Δt`.
`propagate!(...; heisenberg=true)` then returns the genuine Heisenberg-evolved observable
`O(t) = (U^L)† Z_c U^L`. Truncation is applied per Trotter layer (every layer by default,
or every `weight_period`-th layer); the raw operator after each layer is the cost/observable
of record, the kept operator is propagated forward.
