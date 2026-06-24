#!/usr/bin/env julia

# Run one work item from the WeightWhatPP pipeline.
#
# This is a small local/cluster entry point around `src/pipeline.jl`.
# It writes raw `.jls` snapshots to `runs/`, then reads them back to produce the
# lightweight JSON files that can be copied to `data/`.

include(joinpath(@__DIR__, "..", "src", "pipeline.jl"))

experiment = get(ENV, "EXPERIMENT", "rect4x4")
task = lowercase(get(ENV, "TASK", "propagate"))
root = default_runs_root()
mkpath(root)

target_ells = let raw = get(ENV, "TARGET_ELLS", "30,50,80,100")
    [parse(Int, strip(s)) for s in split(raw, ',') if !isempty(strip(s))]
end
cdf_force = lowercase(get(ENV, "FORCE_CDF", get(ENV, "FORCE", "0"))) in ("1", "true", "yes")

items = experiment_work_items(experiment)
storage_experiment = experiment_storage_name(experiment)
idx = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", get(ENV, "ITEM", "0"))) + 1

if idx > length(items)
    error("ITEM/SLURM_ARRAY_TASK_ID=$(idx - 1) is outside the work list for $experiment")
end

item = items[idx]
@info "starting work item" experiment task idx n_items=length(items) item=describe_work_item(item) root

if item.kind == :reference
    if task in ("propagate", "both")
        export_yao_reference!(item.system; experiment = storage_experiment, root)
    elseif task in ("extract", "cdf")
        @info "reference item has no extraction step"
    else
        error("unknown TASK='$task'; expected propagate, extract, cdf, or both")
    end
elseif item.kind == :simulation
    if task == "propagate"
        simulate!(item.system, item.rule; experiment = storage_experiment, root)
    elseif task == "extract"
        extract_metrics!(item.system, item.rule; experiment = storage_experiment, root)
    elseif task == "cdf"
        export_raw_distributions!(item.system, item.rule;
                                  experiment = storage_experiment,
                                  root = root,
                                  ells = target_ells,
                                  force = cdf_force)
    elseif task == "both"
        simulate!(item.system, item.rule; experiment = storage_experiment, root)
        extract_metrics!(item.system, item.rule; experiment = storage_experiment, root)
    else
        error("unknown TASK='$task'; expected propagate, extract, cdf, or both")
    end
else
    error("unknown work item kind: $(item.kind)")
end

@info "finished work item" experiment task idx item=describe_work_item(item)
