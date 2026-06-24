# v3 pipeline loader.  Include this file from the project root with:
#
#     julia --project=. -e 'include("Code/src/pipeline.jl")'

include("systems.jl")
include("truncation.jl")
include("io.jl")
include("trotter.jl")
include("metrics.jl")
include("propagation.jl")
include("reference.jl")
include("experiments.jl")
include("distributions.jl")
