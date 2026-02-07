# src/init_weight.jl
# Initialize random weight matrices for neural networks.

import Random

Random.seed!(1)

"""
	normalize_neg1_to_1(M::Matrix{R})::Matrix{R} where R<:Real

Normalize matrix values from [0, 1) to [-1, 1).

# Examples
```jldoctest
julia> M = [0.0 0.5; 1.0 0.25]
2×2 Matrix{Float64}:
 0.0  0.5
 1.0  0.25
julia> normalize_neg1_to_1(M)
2×2 Matrix{Float64}:
 -1.0   0.0
  1.0  -0.5
```
"""
function normalize_neg1_to_1(M::Matrix{R})::Matrix{R} where R<:Real
	return @. 2 * M - 1
end

"""
	init_rand_weight(dims::Tuple{Int, Int})::Matrix{Float64}

Generate random weight matrices between -1 and 1 for given dims tuples.

# Examples
```jldoctest
julia> dims = 2 => 3
julia> weight_mat = init_rand_weight(dims)
julia> size(weight_mat)
(2, 3)
julia> @assert all(-1 .<= w .<= 1 for w in weight_mats)
```
"""
init_rand_weight::Function = normalize_neg1_to_1 ∘ splat(rand)
