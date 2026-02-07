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
