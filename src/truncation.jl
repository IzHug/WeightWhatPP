# Pauli-weight predicates and the layer-level truncation rule.
#
# This file owns the WEIGHT side of the study: how a Pauli string's weight is
# read from its packed integer, and the concrete `TruncationRule` that turns the
# raw post-layer operator `A_ell` into the kept operator `K_ell`.  Rule
# validation, readable labels, families, the per-term drop predicate, and the
# in-place layer truncation (`apply_rule!` / `truncate!`) all live here so the
# simulation loop can read `truncate!(psum, rule, nq, ell)` and delegate the
# rest.
#
# PauliPropagation encodes a Pauli string as two bits per qubit, with
# least-significant qubit first:
#
#     I = 00, X = 01, Y = 10, Z = 11.

using Printf
using PauliPropagation
using PauliPropagation: PauliSum, getinttype

const PAULI_I = 0x0
const PAULI_X = 0x1
const PAULI_Y = 0x2
const PAULI_Z = 0x3

@inline function pauli_bits(term, q::Int)
    return Int((term >> (2 * q)) & 0x3)
end

"""
    count_weights(term, n_qubits) -> NamedTuple

Return the weight counts used by the truncation rules.  `xy` counts X/Y
slots, `z` counts Z slots, and `xyz = xy + z` counts every non-identity
slot.
"""
function count_weights(term, n_qubits::Int)
    # `term` is the packed integer ID (two bits per qubit), so the weight is read
    # from the bit pattern, not from a list of Pauli letters.  This is a single
    # pass that pools X and Y into `xy` as it goes; counting w_X and w_Y in
    # separate passes would walk the string twice for no benefit, since the rules
    # only use the off-diagonal total w_XY = #X + #Y.  PauliPropagation computes
    # the same quantities branchlessly (bit-plane masks + `count_ones`):
    # `countxy(term) == xy`, `countweight(term) == xyz`, so `z` is
    # `countweight - countxy`.  Delegating to those would remove this loop if
    # profiling ever calls for it; we keep it here for readability.
    xy = 0
    z = 0
    for q in 0:(n_qubits - 1)
        b = pauli_bits(term, q)
        b == PAULI_I && continue
        if b == PAULI_Z
            z += 1
        else
            xy += 1
        end
    end
    return (xy = xy, z = z, xyz = xy + z)
end

"""
    contributes_to_zero(term, n_qubits) -> Bool

`<0|sigma|0>` is non-zero exactly for strings made only of I and Z.
"""
function contributes_to_zero(term, n_qubits::Int)
    for q in 0:(n_qubits - 1)
        b = pauli_bits(term, q)
        (b == PAULI_I || b == PAULI_Z) || return false
    end
    return true
end

# ────────────────────────────────────────────────────────────────────────────
# Concrete truncation-rule model used by the final benchmark.
#
# `TruncationRule` is the runtime representation of one point in the strategy
# sweep: coefficient threshold, XY-weight cap, total-weight cap, optional
# Z-weight cap, and the small per-gate floor passed to PauliPropagation.jl.
# This section also owns rule validation, readable labels, rule families, and
# the predicate that decides whether a term is dropped at a given layer.
# ────────────────────────────────────────────────────────────────────────────

"""
    TruncationRule

One explicit rule for turning the raw post-layer operator `A_ell` into the
kept operator `K_ell`.

Zero disables a threshold.  `c_gate` is the per-gate numerical floor passed
to PauliPropagation and is deliberately included in the rule hash.
"""
Base.@kwdef struct TruncationRule
    coeff_min::Float64 = 0.0
    tau_xy::Int = 0
    tau_xyz::Int = 0
    tau_z::Int = 0
    # Period of the LAYER truncation.  Both the coefficient rule (`coeff_min`) and
    # the weight rules (`tau_*`) fire only on layers with `ell % weight_period == 0`
    # — i.e. every layer when this is `1` (the default).  The name is kept as
    # `weight_period` for run-hash / config compatibility with existing cluster data,
    # but it now gates the whole layer rule, not the weight part alone.
    weight_period::Int = 1
    c_gate::Float64 = 1e-16
    label::String = ""
end

function validate_rule(rule::TruncationRule)
    rule.coeff_min >= 0 || error("coeff_min must be non-negative")
    rule.tau_xy >= 0 || error("tau_xy must be non-negative")
    rule.tau_xyz >= 0 || error("tau_xyz must be non-negative")
    rule.tau_z >= 0 || error("tau_z must be non-negative")
    rule.weight_period >= 1 || error("weight_period must be >= 1")
    rule.c_gate >= 0 || error("c_gate must be non-negative")
    return rule
end

is_physical_noop(rule::TruncationRule) =
    rule.coeff_min == 0.0 &&
    rule.tau_xy == 0 &&
    rule.tau_xyz == 0 &&
    rule.tau_z == 0

function rule_family(rule::TruncationRule)
    rule.weight_period != 1 && rule.tau_xy > 0 && rule.tau_xyz == 0 && return "xspd"
    active_weights = count(!=(0), (rule.tau_xy, rule.tau_xyz, rule.tau_z))
    active_coeff = rule.coeff_min > 0
    active_weights == 0 && active_coeff && return "coeff"
    active_weights == 0 && return "noop"
    active_weights > 1 && return "mixed"
    rule.tau_xy > 0 && return active_coeff ? "xy+coeff" : "xy"
    rule.tau_xyz > 0 && return active_coeff ? "xyz+coeff" : "xyz"
    rule.tau_z > 0 && return active_coeff ? "z+coeff" : "z"
    return "unknown"
end

function readable_rule(rule::TruncationRule)
    !isempty(rule.label) && return rule.label
    parts = String[]
    rule.tau_xy > 0 && push!(parts, "xy$(rule.tau_xy)")
    rule.tau_xyz > 0 && push!(parts, "xyz$(rule.tau_xyz)")
    rule.tau_z > 0 && push!(parts, "z$(rule.tau_z)")
    rule.coeff_min > 0 && push!(parts, "d" * replace(@sprintf("%.0e", rule.coeff_min), "-" => "m"))
    isempty(parts) && push!(parts, "noop")
    rule.weight_period != 1 && push!(parts, "p$(rule.weight_period)")
    rule.c_gate != 1e-16 && push!(parts, "cgate" * replace(@sprintf("%.0e", rule.c_gate), "-" => "m"))
    return join(parts, "_")
end

# The layer truncation (coefficient rule and weight rules alike) fires only on
# layers due by the period; every layer when `weight_period == 1` (the default).
layer_truncation_active(rule::TruncationRule, ell::Int) =
    rule.weight_period == 1 || (ell % rule.weight_period == 0)

function rule_hits(term, coeff::Real, rule::TruncationRule, n_qubits::Int, ell::Int)
    active = layer_truncation_active(rule, ell)
    coeff_hit = active && rule.coeff_min > 0 && abs(Float64(coeff)) < rule.coeff_min
    xy_hit = false
    xyz_hit = false
    z_hit = false
    if active && (rule.tau_xy > 0 || rule.tau_xyz > 0 || rule.tau_z > 0)
        w = count_weights(term, n_qubits)
        xy_hit = rule.tau_xy > 0 && w.xy >= rule.tau_xy
        xyz_hit = rule.tau_xyz > 0 && w.xyz >= rule.tau_xyz
        z_hit = rule.tau_z > 0 && w.z >= rule.tau_z
    end
    weight_hit = xy_hit || xyz_hit || z_hit
    return (
        drop = coeff_hit || weight_hit,
        coeff = coeff_hit,
        xy = xy_hit,
        xyz = xyz_hit,
        z = z_hit,
        weight = weight_hit,
    )
end

# ────────────────────────────────────────────────────────────────────────────
# Layer-level truncation on top of PauliPropagation.jl.
#
# This section does not reimplement the library's gate-by-gate Pauli
# propagation: `simulate!` calls `PauliPropagation.propagate!`, which applies the
# ordered Trotter layer, merges duplicate strings, and applies the small
# per-gate coefficient floor `c_gate`.  The project-specific part starts after a
# full Trotter layer has been propagated.  At that point we apply the benchmark
# rule (`coeff_min`, `tau_xy`, `tau_xyz`, `tau_z`) to the raw end-of-layer Pauli
# sum, delete the rejected terms, and keep a typed list of the dropped
# `(term => coeff)` pairs for diagnostics and later extraction.
#
# The main reason to keep this outside PauliPropagation.jl's built-in
# `truncate!` is observability: the library deletes terms in place, while the
# study needs to know exactly which strings were removed and which threshold
# caused each removal.
# ────────────────────────────────────────────────────────────────────────────

# UInt24 is used by PauliPropagation for 9-12 qubits but is not exported in
# some versions.  Binding it once keeps Julia Serialization round-trips valid.
if !isdefined(PauliPropagation, :UInt24)
    Core.eval(PauliPropagation, :(const UInt24 = $(getinttype(10))))
end

"""
    apply_rule!(psum, rule, n_qubits, ell) -> (dropped, stats)

Drop every term in `psum.terms` that satisfies `rule` at layer `ell`, return
the removed `term => coeff` pairs as `dropped`, and accumulate `stats`.

The implementation collects the kill-list before any mutation to the Dict, so
the iteration is never invalidated by a concurrent `delete!`.  See JuliaLang
issues #15592 / #20623: `Base.Dict` happens to tolerate mid-iteration deletes
on current Julia, but the contract is not guaranteed, and any future internal
rehash on shrink would skip terms silently.
"""
function apply_rule!(psum::PauliSum, rule::TruncationRule, n_qubits::Int, ell::Int)
    validate_rule(rule)
    dropped = Pair{keytype(psum.terms),Float64}[]
    stats = zero_drop_stats()

    for (term, coeff) in psum.terms
        hits = rule_hits(term, coeff, rule, n_qubits, ell)
        hits.drop || continue
        update_drop_stats!(stats, hits)
        push!(dropped, term => Float64(coeff))
    end
    for (term, _) in dropped
        delete!(psum.terms, term)
    end
    return dropped, stats
end

# `truncate!` is the name the simulation loop reads: it is the same in-place
# layer truncation as `apply_rule!`.  We keep `apply_rule!` as the public name
# (external scripts and `truncate_snapshot!` depend on it) and expose
# `truncate!` as an alias so `propagation.jl` reads like the algorithm.
const truncate! = apply_rule!

function update_drop_stats!(stats, hits)
    if hits.coeff && hits.weight
        stats["drop_coeff_and_weight"] += 1
    elseif hits.coeff
        stats["drop_coeff_only"] += 1
    elseif hits.weight
        stats["drop_weight_only"] += 1
    end

    stats["hit_coeff"] += hits.coeff ? 1 : 0
    stats["hit_xy"] += hits.xy ? 1 : 0
    stats["hit_xyz"] += hits.xyz ? 1 : 0
    stats["hit_z"] += hits.z ? 1 : 0
    stats["hit_any_weight"] += hits.weight ? 1 : 0
    stats["hit_multiple_weight"] += count(identity, (hits.xy, hits.xyz, hits.z)) >= 2 ? 1 : 0
    return stats
end

function drop_stats_from_dropped(dropped::AbstractVector, rule::TruncationRule,
                                 n_qubits::Int, ell::Int)
    stats = zero_drop_stats()
    for (term, coeff) in dropped
        hits = rule_hits(term, coeff, rule, n_qubits, ell)
        hits.drop || continue
        update_drop_stats!(stats, hits)
    end
    return stats
end
