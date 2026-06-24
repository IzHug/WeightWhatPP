# Trotter-layer construction for each study system.
#
# This file converts a `StudySystem` into the ordered PauliPropagation.jl gate
# list and matching Heisenberg-picture angle vector used by `simulate!`.

using PauliPropagation

# `trotter_layer(sys)` returns the ordered vector of PauliPropagation gates used
# for one first-order Trotter step.  The order is the order in which the gates
# are applied during Pauli propagation.
#
# For an untilted ring, the layer contains all ZZ coupling rotations first,
# followed by the X-field rotations.  For example, on `chain16` the ZZ gates
# are exactly:
#
#   1:  PauliRotation([:Z, :Z], [1, 2])
#   2:  PauliRotation([:Z, :Z], [2, 3])
#   3:  PauliRotation([:Z, :Z], [3, 4])
#   4:  PauliRotation([:Z, :Z], [4, 5])
#   5:  PauliRotation([:Z, :Z], [5, 6])
#   6:  PauliRotation([:Z, :Z], [6, 7])
#   7:  PauliRotation([:Z, :Z], [7, 8])
#   8:  PauliRotation([:Z, :Z], [8, 9])
#   9:  PauliRotation([:Z, :Z], [9, 10])
#   10: PauliRotation([:Z, :Z], [10, 11])
#   11: PauliRotation([:Z, :Z], [11, 12])
#   12: PauliRotation([:Z, :Z], [12, 13])
#   13: PauliRotation([:Z, :Z], [13, 14])
#   14: PauliRotation([:Z, :Z], [14, 15])
#   15: PauliRotation([:Z, :Z], [15, 16])
#   16: PauliRotation([:Z, :Z], [16, 1])
#
# The remaining gates are `PauliRotation([:X], [1])` through
# `PauliRotation([:X], [16])`.
#
# Notice that PauliPropagation's periodic closing edge is `[16, 1]`, and it is
# the last ZZ gate for `chain16`.
#
# For an untilted open `rect4x4`, sites are laid out row-major:
#
#    1 --(2)--  2 --(4)--  3 --(6)--  4
#    |          |          |          |
#   (1)        (3)        (5)        (7)
#    |          |          |          |
#    5 --(9)--  6 --(11)-  7 --(13)-  8
#    |          |          |          |
#   (8)       (10)       (12)       (14)
#    |          |          |          |
#    9 --(16)- 10 --(18)- 11 --(20)- 12
#    |          |          |          |
#  (15)       (17)       (19)       (21)
#    |          |          |          |
#   13 --(22)- 14 --(23)- 15 --(24)- 16
#
# The numbers in parentheses are the ZZ gate indices.  The exact ZZ gates are:
#
#   1:  PauliRotation([:Z, :Z], [1, 5])
#   2:  PauliRotation([:Z, :Z], [1, 2])
#   3:  PauliRotation([:Z, :Z], [2, 6])
#   4:  PauliRotation([:Z, :Z], [2, 3])
#   5:  PauliRotation([:Z, :Z], [3, 7])
#   6:  PauliRotation([:Z, :Z], [3, 4])
#   7:  PauliRotation([:Z, :Z], [4, 8])
#   8:  PauliRotation([:Z, :Z], [5, 9])
#   9:  PauliRotation([:Z, :Z], [5, 6])
#   10: PauliRotation([:Z, :Z], [6, 10])
#   11: PauliRotation([:Z, :Z], [6, 7])
#   12: PauliRotation([:Z, :Z], [7, 11])
#   13: PauliRotation([:Z, :Z], [7, 8])
#   14: PauliRotation([:Z, :Z], [8, 12])
#   15: PauliRotation([:Z, :Z], [9, 13])
#   16: PauliRotation([:Z, :Z], [9, 10])
#   17: PauliRotation([:Z, :Z], [10, 14])
#   18: PauliRotation([:Z, :Z], [10, 11])
#   19: PauliRotation([:Z, :Z], [11, 15])
#   20: PauliRotation([:Z, :Z], [11, 12])
#   21: PauliRotation([:Z, :Z], [12, 16])
#   22: PauliRotation([:Z, :Z], [13, 14])
#   23: PauliRotation([:Z, :Z], [14, 15])
#   24: PauliRotation([:Z, :Z], [15, 16])
#
# The remaining gates are `PauliRotation([:X], [1])` through
# `PauliRotation([:X], [16])`.
#
# In the tilted case (`sys.g != 0`), the layer inserts the single-qubit Z-field
# rotations after the ZZ couplings and before the X-field rotations.  For
# `rect4x4_h1_g0p5`, this means ZZ gates 1:24, Z gates 25:40, and X gates
# 41:56.

"""
    trotter_layer(sys) -> (layer, thetas)

Return one first-order Trotter layer for the TFIM and the angle vector for
Heisenberg-picture Pauli propagation.
"""
function trotter_layer(sys::StudySystem)
    if sys.g != 0
        layer = tiltedtfitrottercircuit(sys.n_qubits, 1; topology = sys.topology)
    else
        layer = tfitrottercircuit(sys.n_qubits, 1; topology = sys.topology)
    end

    return layer, _thetas(layer, sys.dt; J = sys.J, h = sys.h, g = sys.g)
end

# ──────────────────────────────────────────────────────────────────────────────
# Sign convention (read before editing).
#
# The angles below are negative, θ = -2·{J,h,g}·Δt, and that is correct: 
# the study Hamiltonian is defined with negative signs,
#     H = -J·Σ⟨ij⟩ Z_iZ_j  -  h·Σ_i X_i  -  g·Σ_i Z_i        (report eq. (4)),
# so the forward (Schrödinger) Trotter gate is
#     exp(-i·H_term·Δt) = exp(+i·coupling·Δt·P).
# In PauliPropagation's convention PauliRotation(θ) = exp(-iθ/2·P), matching that gate
# gives  θ = -2·coupling·Δt — exactly the angles set below.
#
# `propagate!(...; heisenberg=true)` then reverses the gate order (`toheisenberg` calls
# `reverse`; the per-gate `_toheisenberg` leaves the angle unchanged) and applies each
# gate's conjugate action U† O U, so it returns the genuine Heisenberg-evolved observable
# O(t) = (U^L)† Z_c U^L for this H.  The Yao statevector reference in reference.jl uses
# the matching Schrödinger angle ϕ = θ, evolving the state under the same H; the two agree
# to ≤1e-15.
# ──────────────────────────────────────────────────────────────────────────────

function _thetas(layer, dt::Real; J::Real, h::Real, g::Real = 0.0)
    thetas = zeros(Float64, countparameters(layer))
    thetas[getparameterindices(layer, PauliRotation, [:Z, :Z])] .= -2 * J * dt
    thetas[getparameterindices(layer, PauliRotation, [:X])] .= -2 * h * dt
    if g != 0
        thetas[getparameterindices(layer, PauliRotation, [:Z])] .= -2 * g * dt
    end
    return thetas
end
