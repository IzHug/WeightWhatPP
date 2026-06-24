# The propagation algorithm.
#
# This file is the simulation loop and nothing else: `simulate!` propagates one
# `(experiment, system, truncation rule)` task Trotter layer by Trotter layer,
# and `truncate_snapshot!` applies the per-layer rule and writes the truncated
# artifact.  The heavy Pauli update is delegated to PauliPropagation.jl; the
# bookkeeping that surrounds the loop — run-directory setup, resume/checkpoint
# discovery, active/done markers, config writing — lives in io.jl and is called
# from here so this file reads like the algorithm.

using PauliPropagation: PauliSum, propagate!

function truncate_snapshot!(root::AbstractString, experiment::AbstractString,
                            sys::StudySystem, rule::TruncationRule,
                            psum::PauliSum, ell::Int)
    raw_n = length(psum.terms)
    dropped, _ = apply_rule!(psum, rule, sys.n_qubits, ell)
    # `dropped` is not used again in this scope, and `write_layer_artifact`
    # writes synchronously before returning, so we serialize the vector
    # directly without a defensive copy.
    write_layer_artifact(truncated_path(root, experiment, sys, rule, ell), dropped)
    return raw_n, length(psum.terms)
end

"""
    simulate!(sys, rule; experiment, root, force=false)

Run one v3 propagation task.  `snapshot_ell.jls` stores the raw operator after
layer `ell` has been propagated, with `snapshot_0.jls` equal to the initial
observable.  `truncated_ell.jls` stores the terms removed from `snapshot_ell`
to prepare the input of layer `ell + 1`.  Either artifact may be a single
`.jls` file or a directory of chunked `.jls` parts if Julia's single-file
serializer cannot represent the object.  Equivalently, `truncated_ell.jls` is
an end-of-layer-ell / beginning-of-layer-(ell+1) truncation record.

This task does not compute `summary.json`.  Run `extract_metrics!` as a separate
task to derive metrics from the serialized layer files.
"""
function simulate!(sys::StudySystem, rule::TruncationRule;
                   experiment::AbstractString = "rect4x4",
                   root::AbstractString = default_runs_root(),
                   force::Bool = false)
    validate_rule(rule)
    run_meta = runtime_metadata()
    prepare_simulation_dir!(root, experiment, sys, rule, run_meta; force = force)

    nl = sys.n_layers

    early = simulation_resume_guard(root, experiment, sys, rule; force = force)
    isnothing(early) || return early

    write_propagation_active!(root, experiment, sys, rule; run_meta = run_meta)

    try
        # `layer` is the ordered gate list (ZZ couplings, then the Z-field if tilted, then the X-field);
        # `thetas` holds the per-gate angles θ = -2·{J,h,g}·Δt — correct for the study's
        # negative-sign Hamiltonian H = -(J·ZZ + h·X + g·Z) (report eq. (4)).  See `_thetas`
        # in trotter.jl for the convention and the historical note on the sign.
        layer, thetas = trotter_layer(sys)
        psum, start_ell = init_or_resume_propagation!(root, experiment, sys, rule)

        for ell in (start_ell + 1):nl
            # Heisenberg picture: `propagate!` conjugates `psum` by the layer and walks the
            # gates in reverse, so the X-field rotations act on O first and the ZZ
            # couplings last — the mirror of the Schrödinger order in which `layer` is
            # built.  Each surviving Pauli's coefficient is its Heisenberg-evolved c_σ.
            propagate!(layer, psum, thetas; min_abs_coeff = rule.c_gate, heisenberg = true)

            # `psum` is the raw operator here; persist that snapshot before truncating.
            write_layer_artifact(snapshot_path(root, experiment, sys, rule, ell), psum.terms)
            # `truncate_snapshot!` drops terms from `psum` *in place* (via `apply_rule!`),
            # so the next layer propagates the kept survivors, not the raw set — this is
            # the end-of-layer truncation policy.  `kept_n == length(psum.terms)` afterwards.
            raw_n, kept_n = truncate_snapshot!(root, experiment, sys, rule, psum, ell)
            @info "layer complete" experiment system=sys.name rule=readable_rule(rule) layer=ell n_raw=raw_n n_kept=kept_n
            GC.gc()
        end

        touch(propagation_done_path(root, experiment, sys, rule))
        return propagation_done_path(root, experiment, sys, rule)
    finally
        rm(propagation_active_path(root, experiment, sys, rule); force = true)
    end
end
