# Filesystem layout, metadata, and robust artifact I/O.
#
# Simulation outputs can be too large for one Julia `.jls` file, so this file
# owns the run-directory schema, atomic JSON writes, single-file serialization,
# chunked layer artifacts, config sidecars, and stable hashes for systems and
# truncation rules.  The rest of the pipeline should use these helpers instead
# of constructing paths or writing artifacts by hand.

using Dates
using JSON3
using Printf
using SHA
using Serialization
using PauliPropagation: PauliSum

const SCHEMA_VERSION = "pds-ppml-v3.0"

default_runs_root() =
    get(ENV, "RUNS_ROOT", joinpath(@__DIR__, "..", "runs"))
default_figures_root() =
    get(ENV, "FIGURES_ROOT", joinpath(@__DIR__, "..", "figures"))

function _atomic_write(path::AbstractString, write_fn)
    mkpath(dirname(path))
    tmp = tempname(dirname(path); cleanup = false)
    try
        open(write_fn, tmp, "w")
        isdir(path) && rm(path; recursive = true, force = true)
        mv(tmp, path; force = true)
    catch
        rm(tmp; force = true)
        rethrow()
    end
    return String(path)
end

_atomic_serialize(path::AbstractString, obj) =
    _atomic_write(path, io -> serialize(io, obj))

_atomic_json_write(path::AbstractString, obj) =
    _atomic_write(path, io -> JSON3.pretty(io, obj))

const CHUNKED_JLS_FORMAT = "pds-ppml-v3-chunked-jls"

function _env_int(name::AbstractString, default::Int)
    value = get(ENV, name, "")
    isempty(value) && return default
    try
        parsed = parse(Int, value)
        parsed >= 0 || error("negative value")
        return parsed
    catch
        @warn "invalid integer environment variable; using default" name value default
        return default
    end
end

_single_jls_max_entries() =
    _env_int("PDS_PPML_SINGLE_JLS_MAX_ENTRIES", Int(typemax(Int32)))

_chunk_jls_entries() =
    max(1, _env_int("PDS_PPML_CHUNK_JLS_ENTRIES", 5_000_000))

_chunkable_artifact(obj) = obj isa AbstractDict || obj isa AbstractVector

function _looks_like_dict_length_overflow(err)
    err isa InexactError || return false
    return occursin("Int32", sprint(showerror, err))
end

function _chunk_name(i::Int)
    return @sprintf("part_%06d.jls", i)
end

function _write_chunk_parts!(tmpdir::AbstractString, obj::AbstractDict,
                             chunk_entries::Int)
    parts = String[]
    K = keytype(obj)
    V = valtype(obj)
    chunk = Dict{K,V}()
    sizehint!(chunk, min(length(obj), chunk_entries))
    n_in_chunk = 0

    for (key, value) in obj
        chunk[key] = value
        n_in_chunk += 1
        if n_in_chunk >= chunk_entries
            part = _chunk_name(length(parts) + 1)
            _atomic_serialize(joinpath(tmpdir, part), chunk)
            push!(parts, part)
            chunk = Dict{K,V}()
            sizehint!(chunk, min(length(obj), chunk_entries))
            n_in_chunk = 0
            GC.gc()
        end
    end

    if n_in_chunk > 0
        part = _chunk_name(length(parts) + 1)
        _atomic_serialize(joinpath(tmpdir, part), chunk)
        push!(parts, part)
    end
    return parts
end

function _write_chunk_parts!(tmpdir::AbstractString, obj::AbstractVector,
                             chunk_entries::Int)
    parts = String[]
    T = eltype(obj)
    chunk = Vector{T}()
    sizehint!(chunk, min(length(obj), chunk_entries))

    for value in obj
        push!(chunk, value)
        if length(chunk) >= chunk_entries
            part = _chunk_name(length(parts) + 1)
            _atomic_serialize(joinpath(tmpdir, part), chunk)
            push!(parts, part)
            chunk = Vector{T}()
            sizehint!(chunk, min(length(obj), chunk_entries))
            GC.gc()
        end
    end

    if !isempty(chunk)
        part = _chunk_name(length(parts) + 1)
        _atomic_serialize(joinpath(tmpdir, part), chunk)
        push!(parts, part)
    end
    return parts
end

function _atomic_serialize_chunked(path::AbstractString, obj; reason::AbstractString)
    mkpath(dirname(path))
    tmpdir = tempname(dirname(path); cleanup = false)
    mkpath(tmpdir)
    try
        chunk_entries = _chunk_jls_entries()
        container_type = obj isa AbstractDict ? "dict" : "vector"
        parts = _write_chunk_parts!(tmpdir, obj, chunk_entries)
        _atomic_json_write(joinpath(tmpdir, "manifest.json"), Dict(
            "format" => CHUNKED_JLS_FORMAT,
            "reason" => String(reason),
            "container_type" => container_type,
            "n_entries" => length(obj),
            "chunk_entries" => chunk_entries,
            "n_parts" => length(parts),
            "parts" => parts,
            "created_at" => string(now()),
            "julia_version" => string(VERSION),
        ))
        ispath(path) && rm(path; recursive = true, force = true)
        mv(tmpdir, path)
    catch
        rm(tmpdir; recursive = true, force = true)
        rethrow()
    end
    return String(path)
end

"""
    write_layer_artifact(path, obj)

Write a layer artifact as the historical single `.jls` file whenever possible.
If the object cannot be represented as one serialized Julia `Dict`/`Vector`
because the entry count exceeds the single-file limit, write `path` as a
directory containing chunked `.jls` parts plus a final `manifest.json`.
"""
function write_layer_artifact(path::AbstractString, obj)
    if _chunkable_artifact(obj) && length(obj) > _single_jls_max_entries()
        return _atomic_serialize_chunked(path, obj; reason = "entry_count_limit")
    end

    try
        return _atomic_serialize(path, obj)
    catch err
        if _chunkable_artifact(obj) && _looks_like_dict_length_overflow(err)
            @warn "single .jls serialization failed; retrying as chunked artifact" path entries=length(obj) exception=sprint(showerror, err)
            return _atomic_serialize_chunked(path, obj; reason = "serialization_int32_overflow")
        end
        rethrow()
    end
end

function layer_artifact_exists(path::AbstractString)
    isfile(path) && return true
    isdir(path) || return false
    manifest = joinpath(path, "manifest.json")
    isfile(manifest) || return false
    obj = try
        JSON3.read(read(manifest, String), Dict)
    catch
        return false
    end
    string(get(obj, "format", "")) == CHUNKED_JLS_FORMAT || return false
    parts = get(obj, "parts", Any[])
    return all(isfile(joinpath(path, string(part))) for part in parts)
end

function read_layer_artifact(path::AbstractString)
    isfile(path) && return deserialize(path)
    if !isdir(path)
        error("layer artifact not found: $path")
    end

    manifest = joinpath(path, "manifest.json")
    isfile(manifest) || error("chunked layer artifact is missing manifest: $path")
    obj = JSON3.read(read(manifest, String), Dict)
    string(get(obj, "format", "")) == CHUNKED_JLS_FORMAT ||
        error("unknown chunked layer artifact format at $path")
    container_type = string(get(obj, "container_type", ""))
    parts = [string(part) for part in get(obj, "parts", Any[])]
    all(isfile(joinpath(path, part)) for part in parts) ||
        error("chunked layer artifact has missing parts: $path")

    if container_type == "dict"
        result = nothing
        for part in parts
            chunk = deserialize(joinpath(path, part))
            chunk isa AbstractDict || error("expected Dict chunk in $path/$part")
            if result === nothing
                result = chunk
            else
                merge!(result, chunk)
            end
        end
        return result === nothing ? Dict() : result
    elseif container_type == "vector"
        result = nothing
        for part in parts
            chunk = deserialize(joinpath(path, part))
            chunk isa AbstractVector || error("expected Vector chunk in $path/$part")
            if result === nothing
                result = chunk
            else
                append!(result, chunk)
            end
        end
        return result === nothing ? Any[] : result
    else
        error("unknown chunked container_type='$container_type' at $path")
    end
end

function system_metadata(sys::StudySystem)
    return Dict(
        "name" => sys.name,
        "n_qubits" => sys.n_qubits,
        "n_layers" => sys.n_layers,
        "dt" => sys.dt,
        "J" => sys.J,
        "h" => sys.h,
        "g" => sys.g,
        "P0_term_hex" => string(sys.P0.term; base = 16),
        "P0_coeff" => Float64(sys.P0.coeff),
        "P0_nqubits" => sys.P0.nqubits,
        "topology_repr" => sprint(show, sys.topology),
    )
end

function rule_metadata(rule::TruncationRule)
    return Dict(
        "label" => readable_rule(rule),
        "family" => rule_family(rule),
        "coeff_min" => rule.coeff_min,
        "tau_xy" => rule.tau_xy,
        "tau_xyz" => rule.tau_xyz,
        "tau_z" => rule.tau_z,
        "weight_period" => rule.weight_period,
        "c_gate" => rule.c_gate,
        "rule_hash" => rule_hash(rule),
    )
end

function rule_hash(rule::TruncationRule)
    validate_rule(rule)
    s = join((
        "coeff=$(repr(rule.coeff_min))",
        "xy=$(rule.tau_xy)",
        "xyz=$(rule.tau_xyz)",
        "z=$(rule.tau_z)",
        "period=$(rule.weight_period)",
        "cgate=$(repr(rule.c_gate))",
    ), ";")
    return bytes2hex(sha256(s))[1:10]
end

function run_dir(root::AbstractString, experiment::AbstractString,
                 sys::StudySystem, rule::TruncationRule)
    return joinpath(root, experiment, sys.name, rule_hash(rule))
end

summary_path(root::AbstractString, experiment::AbstractString,
             sys::StudySystem, rule::TruncationRule) =
    joinpath(run_dir(root, experiment, sys, rule), "summary.json")

extraction_done_path(root::AbstractString, experiment::AbstractString,
                     sys::StudySystem, rule::TruncationRule) =
    joinpath(run_dir(root, experiment, sys, rule), "DONE")

propagation_done_path(root::AbstractString, experiment::AbstractString,
                      sys::StudySystem, rule::TruncationRule) =
    joinpath(run_dir(root, experiment, sys, rule), "PROPAGATION_DONE")

propagation_active_path(root::AbstractString, experiment::AbstractString,
                        sys::StudySystem, rule::TruncationRule) =
    joinpath(run_dir(root, experiment, sys, rule), "PROPAGATION_ACTIVE.json")

extraction_active_path(root::AbstractString, experiment::AbstractString,
                       sys::StudySystem, rule::TruncationRule) =
    joinpath(run_dir(root, experiment, sys, rule), "EXTRACTION_ACTIVE.json")

simulation_config_path(root::AbstractString, experiment::AbstractString,
                       sys::StudySystem, rule::TruncationRule) =
    joinpath(run_dir(root, experiment, sys, rule), "config.json")

snapshot_path(root::AbstractString, experiment::AbstractString,
              sys::StudySystem, rule::TruncationRule, ell::Int) =
    joinpath(run_dir(root, experiment, sys, rule), @sprintf("snapshot_%04d.jls", ell))

truncated_path(root::AbstractString, experiment::AbstractString,
               sys::StudySystem, rule::TruncationRule, ell::Int) =
    joinpath(run_dir(root, experiment, sys, rule), @sprintf("truncated_%04d.jls", ell))

reference_path(root::AbstractString, experiment::AbstractString, sys::StudySystem) =
    joinpath(root, experiment, sys.name, "reference", "yao_overlap.json")

reference_config_path(root::AbstractString, experiment::AbstractString, sys::StudySystem) =
    joinpath(root, experiment, sys.name, "reference", "config.json")

reference_failure_path(root::AbstractString, experiment::AbstractString, sys::StudySystem) =
    joinpath(root, experiment, sys.name, "reference", "failure.json")

simulation_failure_path(root::AbstractString, experiment::AbstractString,
                        sys::StudySystem, rule::TruncationRule) =
    joinpath(run_dir(root, experiment, sys, rule), "failure.json")

function simulation_config_object(experiment::AbstractString, sys::StudySystem,
                                  rule::TruncationRule;
                                  run_meta = runtime_metadata())
    return merge(run_meta, Dict(
        "kind" => "simulation",
        "experiment" => String(experiment),
        "system" => system_metadata(sys),
        "rule" => rule_metadata(rule),
        "layer_file_contract" => Dict(
            "snapshot" => "snapshot_####.jls stores A_ell before layer-ell truncation; it may be a single file or a chunked directory with .jls parts",
            "truncated" => "truncated_####.jls stores D_ell removed before layer ell+1; it may be a single file or a chunked directory with .jls parts",
            "kept" => "K_ell is reconstructed as snapshot_ell - truncated_ell",
        ),
        "metric_extraction" => "summary.json is derived by reading snapshot_ell.jls and truncated_ell.jls, including chunked artifacts when present",
    ))
end

function reference_config_object(experiment::AbstractString, sys::StudySystem;
                                 run_meta = runtime_metadata())
    return merge(run_meta, Dict(
        "kind" => "reference",
        "experiment" => String(experiment),
        "system" => system_metadata(sys),
        "source" => "Yao.jl statevector",
    ))
end

write_simulation_config!(root::AbstractString, experiment::AbstractString,
                         sys::StudySystem, rule::TruncationRule;
                         run_meta = runtime_metadata()) =
    _atomic_json_write(simulation_config_path(root, experiment, sys, rule),
                       simulation_config_object(experiment, sys, rule;
                                                run_meta = run_meta))

write_reference_config!(root::AbstractString, experiment::AbstractString,
                        sys::StudySystem; run_meta = runtime_metadata()) =
    _atomic_json_write(reference_config_path(root, experiment, sys),
                       reference_config_object(experiment, sys;
                                               run_meta = run_meta))

function runtime_metadata()
    project_root = normpath(joinpath(@__DIR__, "..", ".."))
    function git_read(args::AbstractString...)
        cmd = Cmd(["git", "-C", project_root, args...])
        try
            return readchomp(pipeline(cmd, stderr = devnull))
        catch
            return nothing
        end
    end

    git_commit = try
        commit = git_read("rev-parse", "HEAD")
        isnothing(commit) ? "unknown" : commit
    catch
        "unknown"
    end
    git_dirty = try
        status = git_read("status", "--short")
        isnothing(status) ? true : !isempty(status)
    catch
        true
    end
    project_toml = joinpath(project_root, "Project.toml")
    manifest_toml = joinpath(project_root, "Manifest.toml")
    file_hash(path) = isfile(path) ? bytes2hex(sha256(read(path))) : "missing"
    return Dict(
        "schema_version" => SCHEMA_VERSION,
        "created_at" => string(now()),
        "julia_version" => string(VERSION),
        "git_commit" => git_commit,
        "git_dirty" => git_dirty,
        "project_toml_sha256" => file_hash(project_toml),
        "manifest_toml_sha256" => file_hash(manifest_toml),
    )
end

# ────────────────────────────────────────────────────────────────────────────
# Marker, checkpoint, and summary machinery.
#
# These helpers manage the active/done markers, resume-from-checkpoint
# discovery, and `summary.json` writes shared by `simulate!` (propagation.jl)
# and `extract_metrics!` (metrics.jl).  They keep the simulation loop short and
# the extraction phase declarative; nothing here changes the on-disk schema.
# ────────────────────────────────────────────────────────────────────────────

done_path(root::AbstractString, experiment::AbstractString,
          sys::StudySystem, rule::TruncationRule) =
    extraction_done_path(root, experiment, sys, rule)

function summary_object(experiment::AbstractString, sys::StudySystem,
                        rule::TruncationRule, metrics;
                        n_complete::Int, complete::Bool,
                        run_meta = runtime_metadata())
    obj = Dict{String,Any}()
    merge!(obj, run_meta)
    obj["experiment"] = String(experiment)
    obj["complete"] = complete
    obj["n_complete"] = n_complete
    obj["system"] = system_metadata(sys)
    obj["rule"] = rule_metadata(rule)
    for key in METRIC_KEYS
        obj[key] = metrics[key]
    end
    return obj
end

function write_summary!(root::AbstractString, experiment::AbstractString,
                        sys::StudySystem, rule::TruncationRule, metrics;
                        n_complete::Int, complete::Bool,
                        run_meta = runtime_metadata())
    out = summary_path(root, experiment, sys, rule)
    _atomic_json_write(out, summary_object(experiment, sys, rule, metrics;
                                           n_complete = n_complete,
                                           complete = complete,
                                           run_meta = run_meta))
    complete && touch(done_path(root, experiment, sys, rule))
    return out
end

function delete_truncated_terms!(psum::PauliSum, dropped)
    for (term, _) in dropped
        delete!(psum.terms, term)
    end
    return psum
end

function latest_complete_layer(root::AbstractString, experiment::AbstractString,
                               sys::StudySystem, rule::TruncationRule)
    latest = -1
    for ell in 0:sys.n_layers
        layer_artifact_exists(snapshot_path(root, experiment, sys, rule, ell)) || break
        layer_artifact_exists(truncated_path(root, experiment, sys, rule, ell)) || break
        latest = ell
    end
    return latest
end

function _current_hostname()
    return get(ENV, "HOSTNAME", try
        readchomp(`hostname`)
    catch
        "unknown"
    end)
end

function _slurm_job_running(job_id::AbstractString)
    isempty(job_id) && return false
    squeue = Sys.which("squeue")
    squeue === nothing && return false
    try
        return !isempty(strip(readchomp(ignorestatus(`$squeue -h -j $job_id`))))
    catch
        return false
    end
end

function _local_pid_running(pid::AbstractString, hostname::AbstractString)
    isempty(pid) && return false
    hostname == _current_hostname() || return false
    try
        return success(`kill -0 $pid`)
    catch
        return false
    end
end

function propagation_active_metadata(experiment::AbstractString,
                                     sys::StudySystem, rule::TruncationRule;
                                     run_meta = runtime_metadata())
    return merge(run_meta, Dict(
        "kind" => "propagation_active",
        "experiment" => String(experiment),
        "system" => system_metadata(sys),
        "rule" => rule_metadata(rule),
        "pid" => string(getpid()),
        "hostname" => _current_hostname(),
        "slurm_job_id" => get(ENV, "SLURM_JOB_ID", ""),
        "slurm_array_job_id" => get(ENV, "SLURM_ARRAY_JOB_ID", ""),
        "slurm_array_task_id" => get(ENV, "SLURM_ARRAY_TASK_ID", ""),
    ))
end

function extraction_active_metadata(experiment::AbstractString,
                                    sys::StudySystem, rule::TruncationRule;
                                    run_meta = runtime_metadata())
    return merge(run_meta, Dict(
        "kind" => "extraction_active",
        "experiment" => String(experiment),
        "system" => system_metadata(sys),
        "rule" => rule_metadata(rule),
        "pid" => string(getpid()),
        "hostname" => _current_hostname(),
        "slurm_job_id" => get(ENV, "SLURM_JOB_ID", ""),
        "slurm_array_job_id" => get(ENV, "SLURM_ARRAY_JOB_ID", ""),
        "slurm_array_task_id" => get(ENV, "SLURM_ARRAY_TASK_ID", ""),
    ))
end

function write_propagation_active!(root::AbstractString, experiment::AbstractString,
                                   sys::StudySystem, rule::TruncationRule;
                                   run_meta = runtime_metadata())
    return _atomic_json_write(propagation_active_path(root, experiment, sys, rule),
                              propagation_active_metadata(experiment, sys, rule;
                                                          run_meta = run_meta))
end

function write_extraction_active!(root::AbstractString, experiment::AbstractString,
                                  sys::StudySystem, rule::TruncationRule;
                                  run_meta = runtime_metadata())
    return _atomic_json_write(extraction_active_path(root, experiment, sys, rule),
                              extraction_active_metadata(experiment, sys, rule;
                                                         run_meta = run_meta))
end

function active_marker_is_running(path::AbstractString)
    obj = try
        JSON3.read(read(path, String), Dict)
    catch
        return true
    end

    pid = string(get(obj, "pid", ""))
    hostname = string(get(obj, "hostname", ""))
    array_job_id = string(get(obj, "slurm_array_job_id", ""))
    array_task_id = string(get(obj, "slurm_array_task_id", ""))
    if !isempty(array_job_id) && !isempty(array_task_id)
        _slurm_job_running("$(array_job_id)_$(array_task_id)") && return true
        return _local_pid_running(pid, hostname)
    end

    slurm_job_id = string(get(obj, "slurm_job_id", ""))
    _slurm_job_running(slurm_job_id) && return true

    return _local_pid_running(pid, hostname)
end

function propagation_is_active(root::AbstractString, experiment::AbstractString,
                               sys::StudySystem, rule::TruncationRule)
    path = propagation_active_path(root, experiment, sys, rule)
    isfile(path) || return false

    if isfile(propagation_done_path(root, experiment, sys, rule)) &&
       latest_complete_layer(root, experiment, sys, rule) >= sys.n_layers
        rm(path; force = true)
        return false
    end

    active_marker_is_running(path) && return true
    @info "removing stale propagation marker" experiment system=sys.name rule=readable_rule(rule)
    rm(path; force = true)
    return false
end

function extraction_is_active(root::AbstractString, experiment::AbstractString,
                              sys::StudySystem, rule::TruncationRule)
    path = extraction_active_path(root, experiment, sys, rule)
    isfile(path) || return false

    if isfile(extraction_done_path(root, experiment, sys, rule))
        rm(path; force = true)
        return false
    end

    active_marker_is_running(path) && return true
    @info "removing stale extraction marker" experiment system=sys.name rule=readable_rule(rule)
    rm(path; force = true)
    return false
end

# ────────────────────────────────────────────────────────────────────────────
# Simulation prologue helpers.
#
# `simulate!` (propagation.jl) calls these to set up the run directory, decide
# whether the task can be skipped/resumed, and rebuild the working Pauli sum
# from the last checkpoint.  They relocate — without changing — the bookkeeping
# that used to precede the propagation loop.
# ────────────────────────────────────────────────────────────────────────────

function prepare_simulation_dir!(root::AbstractString, experiment::AbstractString,
                                 sys::StudySystem, rule::TruncationRule, run_meta;
                                 force::Bool = false)
    rdir = run_dir(root, experiment, sys, rule)
    if force && isdir(rdir)
        rm(rdir; recursive = true, force = true)
    end
    mkpath(rdir)
    # Write config.json once — first propagation owns the provenance.  Resume
    # paths keep the original timestamp/git_commit; `force=true` deleted the
    # directory above and this branch starts fresh.
    isfile(simulation_config_path(root, experiment, sys, rule)) ||
        write_simulation_config!(root, experiment, sys, rule; run_meta = run_meta)
    return rdir
end

# Returns a path to early-return from `simulate!` when the task is already
# complete or already running, or `nothing` to continue with propagation.
function simulation_resume_guard(root::AbstractString, experiment::AbstractString,
                                 sys::StudySystem, rule::TruncationRule;
                                 force::Bool = false)
    if isfile(propagation_done_path(root, experiment, sys, rule)) && !force
        if latest_complete_layer(root, experiment, sys, rule) >= sys.n_layers
            @info "propagation already complete" experiment system=sys.name rule=readable_rule(rule)
            return propagation_done_path(root, experiment, sys, rule)
        end
        @warn "PROPAGATION_DONE exists but final layer files are incomplete; resuming propagation" experiment system=sys.name rule=readable_rule(rule)
    end

    if !force && propagation_is_active(root, experiment, sys, rule)
        @info "propagation already active; skipping duplicate propagation task" experiment system=sys.name rule=readable_rule(rule)
        return propagation_active_path(root, experiment, sys, rule)
    end

    return nothing
end

# Rebuild the working Pauli sum from the latest checkpoint and return
# `(psum, start_ell)`; layer 0 initialises `A_0 = K_0 = P0` with an empty
# truncated record when no checkpoint exists.
function init_or_resume_propagation!(root::AbstractString, experiment::AbstractString,
                                     sys::StudySystem, rule::TruncationRule)
    nq = sys.n_qubits
    nl = sys.n_layers

    start_ell = -1
    psum = nothing

    complete_layer = latest_complete_layer(root, experiment, sys, rule)
    next_snapshot = complete_layer + 1
    if next_snapshot <= nl && layer_artifact_exists(snapshot_path(root, experiment, sys, rule, next_snapshot))
        psum = PauliSum(nq, read_layer_artifact(snapshot_path(root, experiment, sys, rule, next_snapshot)))
        truncate_snapshot!(root, experiment, sys, rule, psum, next_snapshot)
        start_ell = next_snapshot
        @info "resuming propagation from snapshot" experiment system=sys.name rule=readable_rule(rule) layer=start_ell
    elseif complete_layer >= 0
        psum = PauliSum(nq, read_layer_artifact(snapshot_path(root, experiment, sys, rule, complete_layer)))
        delete_truncated_terms!(psum, read_layer_artifact(truncated_path(root, experiment, sys, rule, complete_layer)))
        start_ell = complete_layer
        @info "resuming propagation from complete layer" experiment system=sys.name rule=readable_rule(rule) layer=start_ell
    end

    if start_ell < 0
        psum = PauliSum(nq, Dict(sys.P0.term => Float64(sys.P0.coeff)))
        write_layer_artifact(snapshot_path(root, experiment, sys, rule, 0), psum.terms)
        # DESIGN.md: A_0 = K_0 = P0 and D_0 = 0.  The rule is not applied at
        # the initial layer, so truncated_0.jls is always an empty vector.
        TT = keytype(psum.terms)
        write_layer_artifact(truncated_path(root, experiment, sys, rule, 0),
                             Pair{TT,Float64}[])
        start_ell = 0
    end

    return psum, start_ell
end
