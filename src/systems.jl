# Physical systems used in the numerical study.
#
# This file defines the code version of Table 1 in the report.  Each catalogue
# entry fixes one benchmark instance: topology, size, Ising parameters, Trotter
# step, number of layers, and initial observable.  The current final study only
# uses periodic rings and open rectangles.

using PauliPropagation: PauliString, staircasetopology, rectangletopology

# ────────────────────────────────────────────────────────────────────────────
# Topology builders: pure graph structure, no `StudySystem` dependency.
#
# A topology is a `Vector{Tuple{Int,Int}}`: an undirected edge list on
# 1-indexed qubits.  It is the ZZ-coupling graph of the TFIM Hamiltonian and
# the only geometry information a `StudySystem` carries.
#
# The current study only uses periodic chains/rings and open rectangles, both
# provided by PauliPropagation.jl.
# ────────────────────────────────────────────────────────────────────────────

# 1D chain/ring: wraps PauliPropagation.staircasetopology
chain_topology(n::Int; periodic::Bool=false) = staircasetopology(n; periodic=periodic)

chain_center(n::Int) = ceil(Int, n / 2)

# 2D rectangular grid — wraps PauliPropagation.rectangletopology
rect_topology(L::Int, W::Int; periodic::Bool=false) = rectangletopology(L, W; periodic=periodic)

# PauliPropagation lays out an (L cols × W rows) grid in row-major order:
#   q = (row - 1) * L + col,   row ∈ 1:W,   col ∈ 1:L.
# For even dimensions this selects the lower-index central site, matching the
# convention used in the previous runs.
function rect_center(L::Int, W::Int)
    row = ceil(Int, W / 2)
    col = ceil(Int, L / 2)
    return (row - 1) * L + col
end

const DEFAULT_DT = 0.01
const DEFAULT_J  = 1.0
const DEFAULT_H  = 3.04438
const DEFAULT_G  = 0.0

"""
    StudySystem

The physical setup for one propagation experiment.  The `name` is part of
the on-disk path, so it must encode any changed physics parameters.
"""
Base.@kwdef struct StudySystem
    name::String
    topology
    n_qubits::Int
    n_layers::Int
    dt::Float64 = DEFAULT_DT
    J::Float64 = DEFAULT_J
    h::Float64 = DEFAULT_H
    g::Float64 = DEFAULT_G
    P0
end

_Z_at(n::Int, q::Int) = PauliString(n, :Z, q)

function make_chain_system(n::Int; periodic::Bool = true, n_layers::Int = 100,
                           dt::Float64 = DEFAULT_DT, J::Float64 = DEFAULT_J,
                           h::Float64 = DEFAULT_H, g::Float64 = DEFAULT_G,
                           name::Union{Nothing,String} = nothing)
    nm = isnothing(name) ? (periodic ? "chain$(n)" : "chain$(n)_open") : name
    return StudySystem(
        name = nm,
        topology = chain_topology(n; periodic),
        n_qubits = n,
        n_layers = n_layers,
        dt = dt,
        J = J,
        h = h,
        g = g,
        P0 = _Z_at(n, chain_center(n)),
    )
end

function make_rect_system(L::Int, W::Int; periodic::Bool = false,
                          n_layers::Int = 100, dt::Float64 = DEFAULT_DT,
                          J::Float64 = DEFAULT_J, h::Float64 = DEFAULT_H,
                          g::Float64 = DEFAULT_G,
                          name::Union{Nothing,String} = nothing)
    n = L * W
    nm = isnothing(name) ? "rect$(L)x$(W)$(periodic ? "_periodic" : "")" : name
    return StudySystem(
        name = nm,
        topology = rect_topology(L, W; periodic),
        n_qubits = n,
        n_layers = n_layers,
        dt = dt,
        J = J,
        h = h,
        g = g,
        P0 = _Z_at(n, rect_center(L, W)),
    )
end

function system_catalogue()
    return Dict(
        "chain16"             => make_chain_system(16; periodic = true, n_layers = 100,
                                                    name = "chain16"),
        "rect4x4"             => make_rect_system(4, 4; n_layers = 100,
                                                  name = "rect4x4"),
        "rect4x4_h1_g0p5"     => make_rect_system(4, 4; n_layers = 100,
                                                  h = 1.0, g = 0.5,
                                                  name = "rect4x4_h1_g0p5"),
        "rect5x5"             => make_rect_system(5, 5; n_layers = 100,
                                                  name = "rect5x5"),
        "rect5x5_h5p158"      => make_rect_system(5, 5; n_layers = 100,
                                                  h = 5.158136,
                                                  name = "rect5x5_h5p158"),
        "rect4x4_dt0p04_l25"  => make_rect_system(4, 4; n_layers = 25,
                                                  dt = 0.04,
                                                  name = "rect4x4_dt0p04_l25"),
        "rect5x5_dt0p04_l25"  => make_rect_system(5, 5; n_layers = 25,
                                                  dt = 0.04,
                                                  name = "rect5x5_dt0p04_l25"),
        "rect11x11_dt0p04_l25" =>
            make_rect_system(11, 11; n_layers = 25, dt = 0.04,
                             name = "rect11x11_dt0p04_l25"),
    )
end
