# Exact statevector reference with Yao.jl.
#
# The Pauli-propagation simulations estimate `<0|O(t)|0>` in the Heisenberg
# picture.  This file builds the matching Schrödinger-picture Trotter circuit
# in Yao.jl and records the same observable exactly by statevector evolution.
# It is used for small and medium systems where a full statevector reference is
# still feasible.

using Yao

const _YAO_PAULI = Dict(:X => Yao.X, :Y => Yao.Y, :Z => Yao.Z)

function _yao_pauli_block(symbols::Vector{Symbol}, qinds, n::Int)
    blocks = [put(n, qinds[i] => _YAO_PAULI[symbols[i]]) for i in eachindex(symbols)]
    return length(blocks) == 1 ? blocks[1] : chain(n, blocks...)
end

function _z_seed_qubit(P0)
    for q in 0:(P0.nqubits - 1)
        pauli_bits(P0.term, q) == PAULI_Z && return q + 1
    end
    error("Yao reference expects P0 to contain a Z seed")
end

function yao_overlap_with_zero(sys::StudySystem)
    layer, thetas_heis = trotter_layer(sys)
    nq = sys.n_qubits
    seed = _z_seed_qubit(sys.P0)

    # Sign convention: the study Hamiltonian carries negative signs,
    # H = -J ΣZZ - h ΣX - g ΣZ (report eq. (4)), so its Schrödinger Trotter step is
    # `exp(-i·H_term·dt) = exp(+i J dt ZZ)`.  trotter.jl already emits the matching
    # angles `θ = -2 J dt` (and analogous for X / Z).  Yao's
    # `rot(block, ϕ) = exp(-i ϕ/2 block)`, so `ϕ = θ = -2 J dt` reproduces it directly.
    gates = [rot(_yao_pauli_block(collect(g.symbols), collect(g.qinds), nq), θ)
             for (g, θ) in zip(layer, thetas_heis)]

    reg = zero_state(nq)
    observable = put(nq, seed => Yao.Z)
    overlaps = Float64[real(expect(observable, reg))]
    for _ in 1:sys.n_layers
        for gate in gates
            apply!(reg, gate)
        end
        push!(overlaps, real(expect(observable, reg)))
    end
    return overlaps
end

function export_yao_reference!(sys::StudySystem;
                               experiment::AbstractString = "rect4x4",
                               root::AbstractString = default_runs_root(),
                               force::Bool = false,
                               max_qubits::Int = 25)
    run_meta = runtime_metadata()
    write_reference_config!(root, experiment, sys; run_meta = run_meta)
    out = reference_path(root, experiment, sys)
    if isfile(out) && !force
        @info "Yao reference already exists" experiment system=sys.name
        return out
    end
    sys.n_qubits <= max_qubits ||
        error("Yao reference for $(sys.name) has $(sys.n_qubits) qubits; max_qubits=$max_qubits")

    @info "computing Yao reference" experiment system=sys.name n_qubits=sys.n_qubits
    overlaps = yao_overlap_with_zero(sys)
    return _atomic_json_write(out, Dict(
        "schema_version" => SCHEMA_VERSION,
        "experiment" => String(experiment),
        "system" => system_metadata(sys),
        "source" => "Yao.jl statevector",
        "overlap_with_zero" => overlaps,
    ))
end
