# Experiment work-list construction.
#
# A named experiment chooses one `StudySystem`, a profile of truncation rules,
# and optionally a Yao reference job.  The functions here are what Slurm array
# scripts and local launchers use to turn names such as `rect4x4` or
# `rect11x11_dt0p04_l25` into concrete `WorkItem`s.  Storage aliases live here
# too, so priority reruns can write into the same run directory as their parent
# experiment.

using Printf

Base.@kwdef struct WorkItem
    kind::Symbol
    system::StudySystem
    rule::Union{Nothing,TruncationRule} = nothing
end

_dlabel(d::Real) = d == 0 ? "d0" : "d" * replace(@sprintf("%.0e", d), "-" => "m")

function unique_rules(rules)
    seen = Set{String}()
    out = TruncationRule[]
    for rule in rules
        h = rule_hash(rule)
        h in seen && continue
        push!(seen, h)
        push!(out, rule)
    end
    return out
end

# Weight cap grid.  The compact 16-qubit profile is the launch profile: five
# k values give 65 simulation rules plus one Yao reference per 16-qubit system.
function k_grid(sys::StudySystem; profile::Symbol = :main)
    if profile == :compact_16 && sys.n_qubits == 16
        return collect(4:8)
    elseif sys.n_qubits == 16
        return collect(5:16)
    elseif sys.n_qubits == 25
        return collect(4:8)
    elseif sys.n_qubits == 121
        return collect(4:8)
    else
        lo = max(3, floor(Int, sys.n_qubits / 2))
        return collect(lo:sys.n_qubits)
    end
end

function delta_grid(; profile::Symbol)
    profile == :largest_delta_only && return (1e-5,)
    return (1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0.0)
end

function rule_grid(sys::StudySystem; profile::Symbol = :main)
    ks = k_grid(sys; profile)
    deltas = delta_grid(; profile)
    rules = TruncationRule[]

    for δ in deltas
        # `tau = none` is the pure coefficient rule.  The δ=0, tau=none
        # no-op is deliberately skipped because it is the near-untruncated
        # Pauli propagation case; Yao is the exact reference instead.
        if δ != 0.0
            push!(rules, TruncationRule(coeff_min = Float64(δ),
                                        label = "coeff_$(_dlabel(δ))"))
        end

        for k in ks
            push!(rules, TruncationRule(tau_xyz = k, coeff_min = Float64(δ),
                                        label = "xyz$(k)_$(_dlabel(δ))"))
        end

        for k in ks
            push!(rules, TruncationRule(tau_xy = k, coeff_min = Float64(δ),
                                        label = "xy$(k)_$(_dlabel(δ))"))
        end
    end

    return unique_rules(rules)
end

const EXPERIMENT_NAMES = (
    "chain16",
    "rect4x4",
    "rect4x4-xy9-12",
    "rect4x4_h1_g0p5",
    "rect5x5",
    "rect5x5_h5p158",
    "rect4x4_dt0p04_l25",
    "rect5x5_dt0p04_l25",
    "rect11x11_dt0p04_l25",
    "rect11x11_dt0p04_l25_d1em04",
)

is_rect4x4_xy9_12_priority(name::AbstractString) =
    String(name) == "rect4x4-xy9-12"

is_rect11x11_d1em04_priority(name::AbstractString) =
    String(name) == "rect11x11_dt0p04_l25_d1em04"

experiment_storage_name(name::AbstractString) =
    is_rect4x4_xy9_12_priority(name) ? "rect4x4" :
    is_rect11x11_d1em04_priority(name) ? "rect11x11_dt0p04_l25" :
    String(name)

const STATUS_EXPERIMENT_NAMES = tuple(
    (name for name in EXPERIMENT_NAMES
     if experiment_storage_name(name) == name)...,
)

function experiment_systems(name::AbstractString)
    cat = system_catalogue()
    name in EXPERIMENT_NAMES ||
        error("unknown experiment '$name'; expected one of $(join(EXPERIMENT_NAMES, ", "))")
    return [cat[experiment_storage_name(name)]]
end

function experiment_profile(name::AbstractString)
    name in EXPERIMENT_NAMES ||
        error("unknown experiment '$name'; expected one of $(join(EXPERIMENT_NAMES, ", "))")
    is_rect4x4_xy9_12_priority(name) && return :priority_rect4x4_xy9_12
    is_rect11x11_d1em04_priority(name) && return :priority_rect11x11_d1em04
    name == "rect11x11_dt0p04_l25" && return :largest_delta_only
    name in ("chain16", "rect4x4", "rect4x4_h1_g0p5", "rect4x4_dt0p04_l25") &&
        return :compact_16
    return :main
end

function priority_rect4x4_xy9_12_items(sys::StudySystem)
    items = WorkItem[]
    for δ in (1e-5, 1e-6), k in 9:12
        push!(items, WorkItem(
            kind = :simulation,
            system = sys,
            rule = TruncationRule(
                tau_xy = k,
                coeff_min = Float64(δ),
                label = "xy$(k)_$(_dlabel(δ))",
            ),
        ))
    end
    return items
end

function priority_rect11x11_d1em04_items(sys::StudySystem)
    δ = 1e-4
    rules = TruncationRule[
        TruncationRule(coeff_min = Float64(δ), label = "coeff_$(_dlabel(δ))"),
    ]
    for k in 4:8
        push!(rules, TruncationRule(
            tau_xyz = k,
            coeff_min = Float64(δ),
            label = "xyz$(k)_$(_dlabel(δ))",
        ))
    end
    for k in 4:8
        push!(rules, TruncationRule(
            tau_xy = k,
            coeff_min = Float64(δ),
            label = "xy$(k)_$(_dlabel(δ))",
        ))
    end
    return [WorkItem(kind = :simulation, system = sys, rule = rule)
            for rule in unique_rules(rules)]
end

function experiment_work_items(name::AbstractString; include_reference::Bool = true)
    systems = experiment_systems(name)
    profile = experiment_profile(name)
    items = WorkItem[]
    for sys in systems
        if profile == :priority_rect4x4_xy9_12
            append!(items, priority_rect4x4_xy9_12_items(sys))
            continue
        elseif profile == :priority_rect11x11_d1em04
            append!(items, priority_rect11x11_d1em04_items(sys))
            continue
        end
        if include_reference && sys.n_qubits <= 25
            push!(items, WorkItem(kind = :reference, system = sys))
        end
        for rule in rule_grid(sys; profile)
            push!(items, WorkItem(kind = :simulation, system = sys, rule = rule))
        end
        if String(name) == "rect4x4"
            append!(items, priority_rect4x4_xy9_12_items(sys))
        elseif String(name) == "rect11x11_dt0p04_l25"
            append!(items, priority_rect11x11_d1em04_items(sys))
        end
    end
    return items
end

function describe_work_item(item::WorkItem)
    item.kind == :reference && return "$(item.system.name) reference"
    return "$(item.system.name) $(readable_rule(item.rule))"
end

"""
    run_work_item!(item; root=default_runs_root(), force=false)

Run one `WorkItem` end to end.  A `:reference` item exports the Yao.jl
statevector reference; a `:simulation` item propagates with `simulate!` and
then derives metrics with `extract_metrics!`.  This is the in-process
equivalent of the `propagate`/`extract`/`both` dispatch in the Slurm launchers,
using the same `experiment = "rect4x4"` default as `simulate!`.
"""
function run_work_item!(item::WorkItem;
                        experiment::AbstractString = "rect4x4",
                        root::AbstractString = default_runs_root(),
                        force::Bool = false)
    if item.kind == :reference
        return export_yao_reference!(item.system; experiment = experiment,
                                     root = root, force = force)
    elseif item.kind == :simulation
        simulate!(item.system, item.rule; experiment = experiment,
                  root = root, force = force)
        return extract_metrics!(item.system, item.rule; experiment = experiment,
                                root = root, force = force)
    else
        error("unknown work item kind: $(item.kind)")
    end
end
