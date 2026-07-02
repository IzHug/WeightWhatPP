# Observables, per-layer metric bookkeeping, and the extraction phase.
#
# This file owns the OBSERVABLE side of the study.  It defines the lightweight
# Pauli-sum measurements (overlap with the zero state, autocorrelation), the
# fixed set of scalar diagnostics that land in `summary.json`, the per-layer
# row writer, and the extraction driver that reads the serialized layer files
# back from disk and derives every metric.  Extraction is deliberately a
# separate task from propagation so it can run after a completed propagation or
# after an OOM/time-limit interruption.

using JSON3
using Serialization
using PauliPropagation: PauliSum

function overlap_zero_terms(terms, n_qubits::Int)
    total = 0.0
    for (term, coeff) in terms
        contributes_to_zero(term, n_qubits) && (total += Float64(coeff))
    end
    return total
end

overlap_zero_pairs(pairs::AbstractVector, n_qubits::Int) =
    overlap_zero_terms(((p.first, p.second) for p in pairs), n_qubits)

"""
    autocorrelation_terms(terms, seed_term) -> Float64

Return the normalized Pauli autocorrelation with the seed observable:

    2^(-n) Tr[A_ell P0]

For Pauli-basis dictionaries this is exactly the coefficient of `seed_term`
when `P0` has unit coefficient, as it does in the current catalogue.
"""
function autocorrelation_terms(terms, seed_term)
    total = 0.0
    for (term, coeff) in terms
        term == seed_term && (total += Float64(coeff))
    end
    return total
end

autocorrelation_pairs(pairs::AbstractVector, seed_term) =
    autocorrelation_terms(((p.first, p.second) for p in pairs), seed_term)

# ────────────────────────────────────────────────────────────────────────────
# Per-layer metric definitions and bookkeeping.
#
# Extraction turns raw snapshots, kept operators, and dropped-term lists into a
# fixed set of scalar diagnostics: sizes, overlaps with the zero state,
# autocorrelation, weight moments, and truncation-hit counts.  This section
# keeps the metric keys and row-writing logic in one place so JSON summaries
# stay schema-stable across propagation runs.
# ────────────────────────────────────────────────────────────────────────────

const METRIC_KEYS = (
    "n_raw",
    "n_kept",
    "n_dropped",
    "overlap_raw",
    "overlap_kept",
    "overlap_dropped",
    # Only `autocorr_raw` is reported.  The autocorrelation is a single
    # Pauli coefficient (c_{P_0}); a kept/dropped split is binary
    # (either the rule touched P_0 or it did not) and carries no
    # information beyond `autocorr_raw` plus the rule definition.
    "autocorr_raw",
    "weight_second_moment_raw",
    "drop_coeff_only",
    "drop_weight_only",
    "drop_coeff_and_weight",
    "hit_coeff",
    "hit_xy",
    "hit_xyz",
    "hit_z",
    "hit_any_weight",
    "hit_multiple_weight",
)

function empty_metrics(n_layers::Int)
    data = Dict{String,Vector{Any}}()
    for key in METRIC_KEYS
        data[key] = Any[nothing for _ in 0:n_layers]
    end
    return data
end

function metrics_from_json(obj)
    data = Dict{String,Vector{Any}}()
    n_layers = try
        Int(obj["system"]["n_layers"])
    catch
        haskey(obj, "n_raw") ? length(obj["n_raw"]) - 1 : 0
    end
    for key in METRIC_KEYS
        data[key] = haskey(obj, key) ? Any[identity(v) for v in obj[key]] :
                    Any[nothing for _ in 0:n_layers]
    end
    return data
end

"""
    weight_second_moment_terms(terms, n_qubits) -> Float64

Return s(ell) = sum_P |c_P(ell)|^2 w(P), using the total Pauli weight
`w(P) = xy(P) + z(P)`.  The caller decides whether `terms` is the raw,
kept, or dropped collection; extraction records the raw end-of-layer snapshot.
"""
function weight_second_moment_terms(terms, n_qubits::Int)
    total = 0.0
    for (term, coeff) in terms
        total += Float64(abs2(coeff)) * count_weights(term, n_qubits).xyz
    end
    return total
end

function zero_drop_stats()
    return Dict(
        "drop_coeff_only" => 0,
        "drop_weight_only" => 0,
        "drop_coeff_and_weight" => 0,
        "hit_coeff" => 0,
        "hit_xy" => 0,
        "hit_xyz" => 0,
        "hit_z" => 0,
        "hit_any_weight" => 0,
        "hit_multiple_weight" => 0,
    )
end

function record_layer!(metrics, ell::Int; raw_n::Int, kept_n::Int,
                       dropped_n::Int, overlap_raw::Float64,
                       overlap_kept::Float64, overlap_dropped::Float64,
                       autocorr_raw::Float64 = 0.0,
                       weight_second_moment_raw::Float64 = 0.0,
                       stats = zero_drop_stats())
    i = ell + 1
    metrics["n_raw"][i] = raw_n
    metrics["n_kept"][i] = kept_n
    metrics["n_dropped"][i] = dropped_n
    metrics["overlap_raw"][i] = overlap_raw
    metrics["overlap_kept"][i] = overlap_kept
    metrics["overlap_dropped"][i] = overlap_dropped
    metrics["autocorr_raw"][i] = autocorr_raw
    metrics["weight_second_moment_raw"][i] = weight_second_moment_raw
    for key in keys(stats)
        metrics[key][i] = stats[key]
    end
    return metrics
end

# ────────────────────────────────────────────────────────────────────────────
# Extraction phase.
#
# Read `snapshot_ell.jls` and `truncated_ell.jls` back from disk, derive all
# metrics, and write `summary.json`.  Each layer artifact can be either the
# historical single `.jls` file or a chunked directory of `.jls` parts.
# ────────────────────────────────────────────────────────────────────────────

function load_partial_summary(path::AbstractString)
    obj = JSON3.read(read(path, String), Dict)
    metrics = metrics_from_json(obj)
    return obj, metrics
end

function summary_has_metric_values(obj, n_layers::Int, through_layer::Int)
    through_layer < 0 && return all(haskey(obj, key) for key in METRIC_KEYS)
    last_i = min(through_layer, n_layers) + 1
    for key in METRIC_KEYS
        haskey(obj, key) || return false
        vals = obj[key]
        length(vals) >= n_layers + 1 || return false
        any(isnothing, vals[1:last_i]) && return false
    end
    return true
end

function _extract_log_every_parts()
    return max(1, _env_int("PDS_PPML_EXTRACT_LOG_EVERY_PARTS", 25))
end

function write_extraction_progress!(root::AbstractString, experiment::AbstractString,
                                    sys::StudySystem, rule::TruncationRule;
                                    layer::Int, artifact::AbstractString,
                                    part::Int, parts::Int, rows::Int)
    path = extraction_active_path(root, experiment, sys, rule)
    obj = if isfile(path)
        try
            Dict{String,Any}(JSON3.read(read(path, String), Dict))
        catch
            extraction_active_metadata(experiment, sys, rule)
        end
    else
        extraction_active_metadata(experiment, sys, rule)
    end
    obj["progress_layer"] = layer
    obj["progress_artifact"] = String(artifact)
    obj["progress_part"] = part
    obj["progress_parts"] = parts
    obj["progress_rows"] = rows
    obj["progress_label"] = parts > 0 ? "$(artifact) $(part)/$(parts)" : String(artifact)
    obj["progress_updated_at"] = string(now())
    _atomic_json_write(path, obj)
    return nothing
end

function _extract_threads()
    requested = max(1, _env_int("PDS_PPML_EXTRACT_THREADS", Threads.nthreads()))
    return min(requested, Threads.nthreads())
end

function _assert_artifact_type(obj, expected::AbstractString, path::AbstractString)
    if expected == "dict"
        obj isa AbstractDict || error("expected Dict artifact at $path")
    elseif expected == "vector"
        obj isa AbstractVector || error("expected Vector artifact at $path")
    else
        error("unknown expected artifact type '$expected'")
    end
    return obj
end

function _layer_artifact_part_paths(path::AbstractString; expected::AbstractString)
    if isfile(path)
        return [(String(path), 1, 1)]
    end
    isdir(path) || error("layer artifact not found: $path")

    manifest = joinpath(path, "manifest.json")
    isfile(manifest) || error("chunked layer artifact is missing manifest: $path")
    obj = JSON3.read(read(manifest, String), Dict)
    string(get(obj, "format", "")) == CHUNKED_JLS_FORMAT ||
        error("unknown chunked layer artifact format at $path")
    container_type = string(get(obj, "container_type", ""))
    container_type == expected ||
        error("expected $expected artifact at $path, got container_type='$container_type'")

    parts = [string(part) for part in get(obj, "parts", Any[])]
    all(isfile(joinpath(path, part)) for part in parts) ||
        error("chunked layer artifact has missing parts: $path")
    n_parts = length(parts)
    return [(joinpath(path, part), i, n_parts) for (i, part) in enumerate(parts)]
end

function _read_metric_chunk(path::AbstractString; expected::AbstractString)
    return _assert_artifact_type(deserialize(path), expected, path)
end

function _mapreduce_metric_chunks(root::AbstractString, experiment::AbstractString,
                                  sys::StudySystem, rule::TruncationRule,
                                  ell::Int, path::AbstractString;
                                  expected::AbstractString,
                                  artifact::AbstractString,
                                  init::Function,
                                  combine::Function,
                                  process_chunk::Function)
    parts = _layer_artifact_part_paths(path; expected)
    n_parts = length(parts)
    log_every = _extract_log_every_parts()
    n_workers = min(_extract_threads(), n_parts)

    if n_workers <= 1
        acc = init()
        rows = 0
        for (part_path, part_i, total_parts) in parts
            chunk = _read_metric_chunk(part_path; expected)
            acc = combine(acc, process_chunk(chunk))
            rows += length(chunk)
            if part_i == 1 || part_i == total_parts || part_i % log_every == 0
                write_extraction_progress!(root, experiment, sys, rule;
                                           layer = ell, artifact = artifact,
                                           part = part_i, parts = total_parts,
                                           rows = rows)
                @info "extracted $(artifact) chunk" experiment system=sys.name rule=readable_rule(rule) layer=ell part=part_i parts=total_parts rows=rows threads=1
            end
            part_i % 8 == 0 && GC.gc()
        end
        return acc
    end

    results = Vector{Any}(undef, n_parts)
    next_i = Threads.Atomic{Int}(0)
    done_parts = Threads.Atomic{Int}(0)
    done_rows = Threads.Atomic{Int}(0)
    progress_lock = ReentrantLock()

    tasks = map(1:n_workers) do _
        Threads.@spawn begin
            while true
                i = Threads.atomic_add!(next_i, 1) + 1
                i > n_parts && break
                part_path, _, total_parts = parts[i]
                chunk = _read_metric_chunk(part_path; expected)
                results[i] = process_chunk(chunk)
                rows = Threads.atomic_add!(done_rows, length(chunk)) + length(chunk)
                done = Threads.atomic_add!(done_parts, 1) + 1
                if done == 1 || done == total_parts || done % log_every == 0
                    lock(progress_lock)
                    try
                        write_extraction_progress!(root, experiment, sys, rule;
                                                   layer = ell, artifact = artifact,
                                                   part = done, parts = total_parts,
                                                   rows = rows)
                        @info "extracted $(artifact) chunks" experiment system=sys.name rule=readable_rule(rule) layer=ell parts_done=done parts=total_parts rows=rows threads=n_workers
                    finally
                        unlock(progress_lock)
                    end
                end
                done % 8 == 0 && GC.gc()
            end
        end
    end
    fetch.(tasks)

    acc = init()
    for result in results
        acc = combine(acc, result)
    end
    return acc
end

function _raw_metrics_chunk(chunk, sys::StudySystem)
    raw_overlap = 0.0
    raw_autocorr = 0.0
    weight_moment = 0.0
    for (term, coeff) in chunk
        c = Float64(coeff)
        contributes_to_zero(term, sys.n_qubits) && (raw_overlap += c)
        term == sys.P0.term && (raw_autocorr += c)
        weight_moment += Float64(abs2(coeff)) * count_weights(term, sys.n_qubits).xyz
    end
    return (length(chunk), raw_overlap, raw_autocorr, weight_moment)
end

_combine_raw_metrics(a, b) = (a[1] + b[1], a[2] + b[2],
                              a[3] + b[3], a[4] + b[4])

function _merge_drop_stats(a, b)
    out = copy(a)
    for (key, value) in b
        out[key] = get(out, key, 0) + value
    end
    return out
end

function _dropped_metrics_chunk(chunk, sys::StudySystem, rule::TruncationRule,
                                ell::Int)
    stats = zero_drop_stats()
    dropped_overlap = 0.0
    for pair in chunk
        term = pair.first
        coeff = pair.second
        contributes_to_zero(term, sys.n_qubits) && (dropped_overlap += Float64(coeff))
        hits = rule_hits(term, coeff, rule, sys.n_qubits, ell)
        hits.drop || continue
        update_drop_stats!(stats, hits)
    end
    return (length(chunk), dropped_overlap, stats)
end

function _combine_dropped_metrics(a, b)
    return (a[1] + b[1], a[2] + b[2], _merge_drop_stats(a[3], b[3]))
end

function raw_metrics_from_disk(root::AbstractString, experiment::AbstractString,
                               sys::StudySystem, rule::TruncationRule,
                               ell::Int)
    snapshot_file = snapshot_path(root, experiment, sys, rule, ell)
    return _mapreduce_metric_chunks(root, experiment, sys, rule, ell, snapshot_file;
        expected = "dict",
        artifact = "raw",
        init = () -> (0, 0.0, 0.0, 0.0),
        combine = _combine_raw_metrics,
        process_chunk = chunk -> _raw_metrics_chunk(chunk, sys),
    )
end

function dropped_metrics_from_disk(root::AbstractString, experiment::AbstractString,
                                   sys::StudySystem, rule::TruncationRule,
                                   ell::Int)
    truncated_file = truncated_path(root, experiment, sys, rule, ell)
    return _mapreduce_metric_chunks(root, experiment, sys, rule, ell, truncated_file;
        expected = "vector",
        artifact = "dropped",
        init = () -> (0, 0.0, zero_drop_stats()),
        combine = _combine_dropped_metrics,
        process_chunk = chunk -> _dropped_metrics_chunk(chunk, sys, rule, ell),
    )
end

function record_layer_from_disk!(root::AbstractString, experiment::AbstractString,
                                 sys::StudySystem, rule::TruncationRule, metrics,
                                 ell::Int, run_meta; complete::Bool = false)
    @info "extracting layer from disk" experiment system=sys.name rule=readable_rule(rule) layer=ell
    write_extraction_progress!(root, experiment, sys, rule;
                               layer = ell, artifact = "starting",
                               part = 0, parts = 0, rows = 0)
    raw_n, raw_overlap, raw_autocorr, weight_moment =
        raw_metrics_from_disk(root, experiment, sys, rule, ell)
    dropped_n, dropped_overlap, stats =
        dropped_metrics_from_disk(root, experiment, sys, rule, ell)

    record_layer!(metrics, ell;
        raw_n = raw_n,
        kept_n = raw_n - dropped_n,
        dropped_n = dropped_n,
        overlap_raw = raw_overlap,
        overlap_kept = raw_overlap - dropped_overlap,
        overlap_dropped = dropped_overlap,
        autocorr_raw = raw_autocorr,
        weight_second_moment_raw = weight_moment,
        stats = stats,
    )
    write_summary!(root, experiment, sys, rule, metrics;
                   n_complete = ell, complete = complete, run_meta = run_meta)
    return raw_n, raw_n - dropped_n
end

"""
    extract_metrics!(sys, rule; experiment, root, force=false)

Read `snapshot_ell.jls` and `truncated_ell.jls` from disk, derive all metrics,
and write `summary.json`.  Each layer artifact can be either the historical
single `.jls` file or a chunked directory of `.jls` parts.  This is
intentionally a separate task from propagation so it can be launched after a
completed propagation or after an OOM/time-limit interruption.
"""
function extract_metrics!(sys::StudySystem, rule::TruncationRule;
                          experiment::AbstractString = "rect4x4",
                          root::AbstractString = default_runs_root(),
                          force::Bool = false)
    validate_rule(rule)
    rdir = run_dir(root, experiment, sys, rule)
    mkpath(rdir)
    run_meta = runtime_metadata()
    # config.json is owned by the propagation step and records the propagation
    # provenance (git commit, julia version, schema).  We only write it here
    # if it is missing — usually because extraction is being run against a
    # legacy directory whose propagation predated the current schema.
    isfile(simulation_config_path(root, experiment, sys, rule)) ||
        write_simulation_config!(root, experiment, sys, rule; run_meta = run_meta)

    spath = summary_path(root, experiment, sys, rule)
    if propagation_is_active(root, experiment, sys, rule)
        @info "propagation active; extraction skipped" experiment system=sys.name rule=readable_rule(rule)
        return spath
    end

    if !force && extraction_is_active(root, experiment, sys, rule)
        @info "extraction already active; skipping duplicate extraction task" experiment system=sys.name rule=readable_rule(rule)
        return extraction_active_path(root, experiment, sys, rule)
    end

    write_extraction_active!(root, experiment, sys, rule; run_meta = run_meta)

    try
        metrics = empty_metrics(sys.n_layers)
        start_ell = 0

        if isfile(spath) && !force
            obj, loaded_metrics = load_partial_summary(spath)
            recorded_through = Int(get(obj, "n_complete", -1))
            schema_current = summary_has_metric_values(obj, sys.n_layers, recorded_through)
            if get(obj, "complete", false) == true &&
               isfile(done_path(root, experiment, sys, rule)) &&
               schema_current
                @info "extraction already complete" experiment system=sys.name rule=readable_rule(rule)
                return spath
            end
            if schema_current
                metrics = loaded_metrics
                start_ell = max(0, recorded_through + 1)
            else
                @info "refreshing summary with current metric schema" experiment system=sys.name rule=readable_rule(rule)
                metrics = empty_metrics(sys.n_layers)
                start_ell = 0
            end
        end

        last = latest_complete_layer(root, experiment, sys, rule)
        last >= 0 || error("no complete snapshot/truncated layer pair found for $(sys.name) $(readable_rule(rule))")

        for ell in start_ell:last
            record_layer_from_disk!(root, experiment, sys, rule, metrics, ell, run_meta;
                                    complete = ell == sys.n_layers)
            @info "extracted layer" experiment system=sys.name rule=readable_rule(rule) layer=ell
            GC.gc()
        end

        if start_ell > last
            write_summary!(root, experiment, sys, rule, metrics;
                           n_complete = last, complete = last == sys.n_layers,
                           run_meta = run_meta)
        end

        return spath
    finally
        rm(extraction_active_path(root, experiment, sys, rule); force = true)
    end
end
