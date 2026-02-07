# src/propagate/fore/forward_propagate.jl
# Functions for forward propagation through a neural network

include("../../relu.jl")

"""
	advance_layer(
		layer::L,
		weights::Matrix{R}
	)::Matrix{R} where {L<:AbstractMatrix, R<:Real}

Advance the given layer by multiplying with weights and applying `ReLU` activation.

# Examples
```jldoctest
julia> using Grok: advance_layer
julia> layer = [0.5 0.2 0.1];
julia> weights = [0.4 0.6; 0.3 0.9; 0.8 0.2];
julia> advance_layer(layer, weights)
2×1 Matrix{Float64}:
 0.38
 0.39
```
"""
function advance_layer(
	layer::L,
	weights::Matrix{R}
)::Matrix{R} where {L<:AbstractMatrix, R<:Real}
	
	return relu.(layer * weights)
end
