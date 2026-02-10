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

"""
	forward_propagate(
		layer_in::Row,
		weight_mats::Vector{Matrix{R}}
	)::Vector{Matrix{R}} where {Row<:AbstractVector{<:Real}, R<:Real}

Perform forward propagation through the network.

# Examples
```jldoctest
julia> using Grok: forward_propagate
julia> layer_in = [0.5, 0.2, 0.1];
julia> weight_mats = [ [0.4 0.6; 0.3 0.9; 0.8 0.2], [0.5; 0.7] ];
julia> forward_propagate(layer_in, weight_mats)
2-element Vector{Matrix{Float64}}:
 [0.38; 0.39]
 [0.6119999999999999]
```
"""
function forward_propagate(
	layer_in::Row,
	Ws::Arrays
)::Tuple{Vector{Vector{R}}, Vector{R}} where {Row<:AbstractVector{<:Real}, R<:Real, Arrays<:AbstractVector{<:AbstractArray{R}}}

	# Ensure every weight is a 2D matrix (convert 1D vectors to column matrices)
	mats::Vector{Matrix{R}} = [ndims(w) == 1 ? reshape(w, :, 1) : Array(w) for w in Ws]

	# compute intermediate layer matrices (each result is a 2D matrix)
	intermediate_mats = accumulate(advance_layer, mats[begin:end-1]; init = layer_in')

	# final layer output matrix
	layer_out_mat = if intermediate_mats |> isempty
		layer_in' * mats[end]
	else
		intermediate_mats[end] * mats[end]
	end

	# convert matrices to 1D column vectors to match test expectations
	intermediate_layers = vec.(intermediate_mats)
	layer_out = vec(layer_out_mat)

	return intermediate_layers, layer_out
end
