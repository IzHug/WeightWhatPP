# Distribution extracts for Pauli-sum populations.
#
# The propagation tasks write large raw snapshots (`A_ell`) and dropped-term
# lists (`T_ell`) as Julia artifacts.  This file reads those artifacts and
# turns them into small JSON-ready histograms of Pauli weight and coefficient
# magnitude.  These compact summaries are used for CDF/diagnostic plots without
# loading the full Pauli sum in Python.

using JSON3
using Serialization

"""
    extract_raw_distributions(experiment, sys, rule, ell; root, n_coeff_bins, coeff_log_range)

Walk the raw post-gate operator ``A_ell`` (the end-of-layer Pauli sum, before
the per-layer truncation is applied) for one (experiment, system, rule) at a
given Trotter layer ``ell`` and produce compact histograms of:
  - Pauli weights ``w_XYZ`` in `0..n_qubits` (one bucket per integer)
  - ``log10|c_sigma|`` in `n_coeff_bins` log-spaced bins over the configurable
    range, by default `[10^-16, 10^0]`.

Reads `snapshot_NNNN.jls` (which stores ``A_ell``) directly.  Returns a
JSON-ready dictionary with a tiny payload (~ `n_qubits + n_coeff_bins` ints).
Returns `nothing` when the snapshot file is missing.
"""
function extract_raw_distributions(experiment::AbstractString,
                                   sys::StudySystem,
                                   rule::TruncationRule,
                                   ell::Int;
                                   root::AbstractString=default_runs_root(),
                                   n_coeff_bins::Int=120,
                                   coeff_log_range::Tuple{Float64,Float64}=(-16.0, 0.0))
    snap_path = snapshot_path(root, experiment, sys, rule, ell)
    layer_artifact_exists(snap_path) || return nothing

    nq = sys.n_qubits
    log_lo, log_hi = coeff_log_range
    log_step = (log_hi - log_lo) / n_coeff_bins

    weight_hist = zeros(Int, nq + 1)
    coeff_hist  = zeros(Int, n_coeff_bins)
    joint_hist  = zeros(Int, nq + 1, n_coeff_bins)
    n_raw = 0

    for (term, c) in layer_artifact_entries(snap_path; expected = "dict")
        w = count_weights(term, nq).xyz
        weight_hist[w + 1] += 1

        ac = abs(Float64(c))
        if ac > 0
            log_ac = log10(ac)
            i = floor(Int, (log_ac - log_lo) / log_step) + 1
            i = clamp(i, 1, n_coeff_bins)
            coeff_hist[i] += 1
            joint_hist[w + 1, i] += 1
        end
        n_raw += 1
    end

    max_w = -1
    for w in nq:-1:0
        if weight_hist[w + 1] > 0
            max_w = w
            break
        end
    end

    return Dict(
        "ell"               => ell,
        "n_raw"             => n_raw,
        "n_qubits"          => nq,
        "weight_histogram"  => weight_hist,
        "max_weight"        => max_w,
        "coeff_hist"        => coeff_hist,
        # Joint histogram [w, log|c| bin]: count of strings with that pair.
        # Stored as Vector{Vector{Int}} (row per weight) for JSON serialisation.
        "joint_hist"        => [joint_hist[w, :] for w in 1:(nq + 1)],
        "coeff_log_lo"      => log_lo,
        "coeff_log_hi"      => log_hi,
        "coeff_n_bins"      => n_coeff_bins,
        "kind"              => "raw",
    )
end

"""
    extract_truncated_distributions(experiment, sys, rule, ell; root, n_coeff_bins, coeff_log_range)

Walk the truncated set ``T_ell`` (the strings dropped at layer ``ell``) for one
(experiment, system, rule) and produce the same compact (weight, log|c|)
histograms as `extract_raw_distributions`.

Reads `truncated_NNNN.jls`, which stores a `Vector{Pair{TermType, Float64}}` of
the dropped (string, coefficient) pairs.  Returns `nothing` when the file is
missing.
"""
function extract_truncated_distributions(experiment::AbstractString,
                                         sys::StudySystem,
                                         rule::TruncationRule,
                                         ell::Int;
                                         root::AbstractString=default_runs_root(),
                                         n_coeff_bins::Int=120,
                                         coeff_log_range::Tuple{Float64,Float64}=(-16.0, 0.0))
    trunc_path = truncated_path(root, experiment, sys, rule, ell)
    layer_artifact_exists(trunc_path) || return nothing

    nq = sys.n_qubits
    log_lo, log_hi = coeff_log_range
    log_step = (log_hi - log_lo) / n_coeff_bins

    weight_hist = zeros(Int, nq + 1)
    coeff_hist  = zeros(Int, n_coeff_bins)
    joint_hist  = zeros(Int, nq + 1, n_coeff_bins)
    n_truncated = 0

    for (term, c) in layer_artifact_entries(trunc_path; expected = "vector")
        w = count_weights(term, nq).xyz
        weight_hist[w + 1] += 1

        ac = abs(Float64(c))
        if ac > 0
            log_ac = log10(ac)
            i = floor(Int, (log_ac - log_lo) / log_step) + 1
            i = clamp(i, 1, n_coeff_bins)
            coeff_hist[i] += 1
            joint_hist[w + 1, i] += 1
        end
        n_truncated += 1
    end

    max_w = -1
    for w in nq:-1:0
        if weight_hist[w + 1] > 0
            max_w = w
            break
        end
    end

    return Dict(
        "ell"               => ell,
        "n_truncated"       => n_truncated,
        "n_qubits"          => nq,
        "weight_histogram"  => weight_hist,
        "max_weight"        => max_w,
        "coeff_hist"        => coeff_hist,
        "joint_hist"        => [joint_hist[w, :] for w in 1:(nq + 1)],
        "coeff_log_lo"      => log_lo,
        "coeff_log_hi"      => log_hi,
        "coeff_n_bins"      => n_coeff_bins,
        "kind"              => "truncated",
    )
end

function _chunk_manifest(path::AbstractString)
    manifest = joinpath(path, "manifest.json")
    isfile(manifest) || error("chunked layer artifact is missing manifest: $path")
    obj = JSON3.read(read(manifest, String), Dict)
    string(get(obj, "format", "")) == CHUNKED_JLS_FORMAT ||
        error("unknown chunked layer artifact format at $path")
    return obj
end

function layer_artifact_entries(path::AbstractString; expected::AbstractString)
    if isfile(path)
        obj = deserialize(path)
        return obj
    end
    isdir(path) || error("layer artifact not found: $path")

    manifest = _chunk_manifest(path)
    container_type = string(get(manifest, "container_type", ""))
    container_type == expected ||
        error("expected $expected artifact at $path, got container_type='$container_type'")
    parts = [string(part) for part in get(manifest, "parts", Any[])]
    all(isfile(joinpath(path, part)) for part in parts) ||
        error("chunked layer artifact has missing parts: $path")

    return Iterators.flatten((deserialize(joinpath(path, part)) for part in parts))
end

"""
    export_raw_distributions!(sys, rule; experiment, root, ells, force)

Run `extract_raw_distributions` AND `extract_truncated_distributions` for each
`ell` in `ells` and write both into `<run_dir>/cdf_layers.json` under the
`layers` (= ``A_ell``) and `truncated_layers` (= ``T_ell``) keys.

`force = false` (default) skips a run whose JSON already exists.  Returns the
output path, or `nothing` if no layers produced any data.
"""
function export_raw_distributions!(sys::StudySystem,
                                   rule::TruncationRule;
                                   experiment::AbstractString="chain16",
                                   root::AbstractString=default_runs_root(),
                                   ells::Vector{Int}=[30, 50, 100],
                                   force::Bool=false)
    rdir = run_dir(root, experiment, sys, rule)
    isdir(rdir) || return nothing
    out_path = joinpath(rdir, "cdf_layers.json")
    out = if isfile(out_path) && !force
        JSON3.read(read(out_path, String), Dict{String,Any})
    else
        Dict{String,Any}(
            "system"           => sys.name,
            "rule"             => rule_metadata(rule),
            "experiment"       => String(experiment),
            "ells"             => Int[],
            "layers"           => Dict{String,Any}(),
            "truncated_layers" => Dict{String,Any}(),
        )
    end

    existing_ells = Int[]
    for value in get(out, "ells", Any[])
        parsed = value isa Integer ? Int(value) : tryparse(Int, string(value))
        parsed === nothing || push!(existing_ells, parsed)
    end
    out["ells"] = sort!(unique!(vcat(existing_ells, ells)))
    haskey(out, "layers") || (out["layers"] = Dict{String,Any}())
    haskey(out, "truncated_layers") || (out["truncated_layers"] = Dict{String,Any}())

    for ell in ells
        key = string(ell)
        if !force && haskey(out["layers"], key) &&
           haskey(out["truncated_layers"], key)
            continue
        end

        wrote_any = false
        if force || !haskey(out["layers"], key)
            d_raw = extract_raw_distributions(experiment, sys, rule, ell; root=root)
            if d_raw !== nothing
                out["layers"][key] = d_raw
                wrote_any = true
            end
        end

        if force || !haskey(out["truncated_layers"], key)
            d_trunc = extract_truncated_distributions(experiment, sys, rule, ell; root=root)
            if d_trunc !== nothing
                out["truncated_layers"][key] = d_trunc
                wrote_any = true
            end
        end

        if wrote_any
            out["updated_at_layer"] = ell
            out["complete_layers"] = [
                layer for layer in out["ells"]
                if haskey(out["layers"], string(layer)) &&
                   haskey(out["truncated_layers"], string(layer))
            ]
            _atomic_json_write(out_path, out)
            @info "updated CDF/joint layer artifact" experiment system=sys.name rule=readable_rule(rule) layer=ell out_path
        end
    end

    if isempty(out["layers"]) && isempty(out["truncated_layers"])
        return nothing
    end
    _atomic_json_write(out_path, out)
    return out_path
end
