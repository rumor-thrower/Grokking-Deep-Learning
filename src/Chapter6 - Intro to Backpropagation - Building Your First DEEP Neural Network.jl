### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ c4ac87fe-fe8f-11f0-bd44-e7be41300a4b
md"""
# Creating a Matrix or Two in Julia
"""

# ╔═╡ 941e115e-307c-4aec-a6d6-5f90866ecc3e
function get_inputs_and_goals()::Tuple{RowSlices, Base.Generator}
	streetlights = BitMatrix([ 1 0 1 ;
							   0 1 1 ;
							   0 0 1 ;
							   1 1 1 ;
							   0 1 1 ;
							   1 0 1 ])
	
	walk_vs_stop = BitVector([ 0, 1, 0, 1, 1, 0 ])

	inputs::RowSlices = eachrow(streetlights)
	goals::Base.Generator = (bool for bool in walk_vs_stop)

	return (inputs, goals)
end

# ╔═╡ 9961822e-8687-49fe-8e76-2d456334ee57
import LinearAlgebra

# ╔═╡ 34cabbbf-a02f-410d-9e0f-43493eec46f1
begin
	"""
		fit_factory(
			input::Row,
			goal::Bool,
			alpha::R
		)::Function where {Row<:AbstractVector{<:Real}, R<:Real}
	
	Returns a function that fits a neural network for a single input-goal pair with learning rate `alpha`.
	"""
	function fit_factory(
		input::Row,
		goal::Bool,
		alpha::R
	)::Function where {Row<:AbstractVector{<:Real}, R<:Real}
		
		"""
			fit(weights::W, epochs_left::Int)::W where W<:Vector{<:Real}
		
		Fits the neural network weights for the given number of epochs.
		"""
	function fit(weights::W, epochs_left::Int)::W where W<:Vector{<:Real}
			
			if epochs_left < 1
				return weights
			end
	
			R2 = eltype(W)
		
			pred::R2 = LinearAlgebra.dot(input, weights)
			delta::R2 = pred - goal
			error::R2 = delta ^ 2
			weights::W -= alpha * input * delta
		
			@debug "Status:" input error pred
			
			return fit(weights, epochs_left - 1)
		end
		
		function fit(weight_mats, epochs_left::Int)
			
			subsequent_layers = forward_propagate(input, weight_mats)
			
			L_out = subsequent_layers[end]
			pred_f, ΔL_out = calc_loss(L_out, goal)
			
			ΔWs = back_propagate(input, subsequent_layers, ΔL_out, weight_mats)
			weight_mats .-= alpha .* ΔWs
			
			# L_out_error = sum(abs2, ΔL_out)
			return weight_mats
		end

		return fit
	end
	
	"""
		fit_factory(
			(input, goal)::Tuple{Row, Bool},
			alpha::R
		)::Function where {Row<:AbstractVector{<:Real}, R<:Real}
	
	Variant of `fit_factory` that takes a sample tuple instead of separate input and goal.
	"""
	function fit_factory(
		(input, goal)::Tuple{Row, Bool},
		alpha::R
	)::Function where {Row<:AbstractVector{<:Real}, R<:Real}
		
		return fit_factory(input, goal, alpha)
	end
end

# ╔═╡ 226b0583-10a8-4fe7-a6c2-9fbd5136a4f1
let (inputs, goals) = get_inputs_and_goals()
	input, goal = first.([inputs, goals])
	
	fit::Function = fit_factory(input, goal, .1)
	
	fit([.5, .48, -.7], 20)
end

# ╔═╡ 20f1e96c-613d-42f4-928a-3c4f033263e2
md"""
## Building Our Neural Network
"""

# ╔═╡ 0ecea74e-0443-4d1a-9843-b1a8bee23eb5
let
	a = [0, 1, 2, 1]
	b = [2, 2, 2, 3]
	
	@assert a .* b == [0, 2, 4, 3] # elementwise multiplication
	@assert a .+ b == [2, 3, 4, 4] # elementwise addition
	@assert a .* 0.5 == [.0, .5, 1.0, .5] # vector-scalar multiplication
	@assert a .+ 0.5 == [.5, 1.5, 2.5, 1.5] # vector-scalar addition
end

# ╔═╡ e2ea141a-560e-4f68-be97-44dbfc749a0f
md"""
# Learning the whole dataset!
"""

# ╔═╡ 309d4b06-274a-4390-83bf-c75cd722202e
"""
	train_network(
		initial_weights::Vector{R},
		alpha::R,
		max_epoch::Int
	)::Vector{R} where R<:Real

Trains a neural network with initial weights `initial_weights`, learning rate `alpha`, and for `max_epoch` epochs over the entire dataset.
"""
function train_network(
	initial_weights::W,
	alpha::R,
	max_epoch::Int
)::W where {R<:Real, W<:Vector{R}}
	
	# sample = (input, goal)
	samples = get_inputs_and_goals()
	
	# fitter = fit_factory(sample, alpha)
	sample_to_fitter::Function = Base.Fix2(fit_factory, alpha)
	
	fitters_per_sample::Vector{Function} = zip(samples...) .|> sample_to_fitter
	
	function fit_weights_once(weights::W, fit::Function)::W where W<:Vector{<:Real}
		return fit(weights, 1)
	end
	
	function batch_update(weights::W, epoch::Int)::W where W<:Vector{<:Real}
		@info "Begin $epoch th epoch"
		# update weight on all sample
		# weights = fit(weights, 1)
		return reduce(fit_weights_once, fitters_per_sample; init = weights)
	end
	
	return reduce(batch_update, 1:max_epoch; init = initial_weights)
end

# ╔═╡ 10369220-f87d-4f0b-9477-74ac7121b2f1
train_network([.5, .48, -.7], .1, 40)

# ╔═╡ 9309ea1d-de47-4a98-9989-c3cfe50dc0a8
md"""
# Our First "Deep" Neural Network
"""

# ╔═╡ 3c2fa6d0-0e3b-4f7e-925b-1486e52b6388
import Random

# ╔═╡ 18836503-681e-41bf-b214-3db0af8d14a2
Random.seed!(1)

# ╔═╡ 43386f27-ada9-483d-bc18-9c3ddbff0503
"""
	normalize_neg1_to_1(M::Matrix{R})::Matrix{R} where R<:Real

Normalize matrix values from [0, 1) to [-1, 1).
"""
function normalize_neg1_to_1(M::Matrix{R})::Matrix{R} where R<:Real
	return @. 2 * M - 1
end

# ╔═╡ ce062beb-067f-4f5b-b81c-f310a8938630
"""
	init_rand_weight(sizes::Vector{Tuple{Int, Int}})::Vector{Matrix{Float64}}

Generate random weight matrices between -1 and 1 for given size tuples.
"""
init_rand_weight::Function = normalize_neg1_to_1 ∘ splat(rand)

# ╔═╡ 89f3c7c3-0c68-4c56-860f-eba8a5d7428d
module ReLU

export activate, deriv_factory

"""
	activate(x::R)::R where R<:Real

Apply `ReLU` activation function.

# Returns
Maximum of input `x` and zero for input type `R`.
"""
function activate(x::R)::R where R<:Real
	return max(zero(R), x)
end

"""
	deriv_factory()::Function

# Returns
Derivative function of `ReLU`. At `x = 0`, throws `ArgumentError`.

# Explanation
The derivative of `ReLU` is:
- `0` for `x < 0`
- `1` for `x > 0`
- undefined at `x = 0` (since the left-hand limit and right-hand limit do not agree).

# Example
```julia
using ReLU
deriv::Function = deriv_factory()
deriv(-.5)
# 0
deriv(.5)
# 1
deriv(.0)
# throws ArgumentError
```
"""
deriv_factory()::Function =
	function relu2deriv(x::R) where R<:Real
		handle_zero(x::R) = throw(ArgumentError("The derivative at x = $x is undefined"))
		handler::Function = x |> iszero ? handle_zero : R ∘ !signbit
		return handler(x)
	end

"""
	deriv_factory(fallback::R)::Function where R<:Real

Get the derivative function of `ReLU` which uses `fallback` at `x = 0`.

# Example
```julia
using ReLU
deriv::Function = deriv_factory(.0)
deriv(-.5)
# 0
deriv(.5)
# 1
deriv(.0)
# 0.0
```
"""
function deriv_factory(fallback::R)::Function where R<:Real
	relu_deriv(x::R)::R = ifelse(x |> iszero, fallback, !signbit(x))
end

end

# ╔═╡ 8514e517-debe-4381-a7c1-3dba9dc107ee
"""
	advance_layer(
		layer::L,
		weights::Matrix{R}
	)::Matrix{R} where {L<:AbstractMatrix, R<:Real}

Advance the given layer by multiplying with weights and applying `ReLU` activation.

# Arguments
- `layer::L`: The input layer as a matrix.
    - **Expected**: a row vector(1 x `n`), from `RowSlices` of the data maxtrix, of type `LinearAlgebra.Adjoint`.
        - `n` is the number of neurons in the layer.
- `weights::Matrix{R}`: The weight matrix to apply.
"""
function advance_layer(
	layer::L,
	weights::Matrix{R}
)::Matrix{R} where {L<:AbstractMatrix, R<:Real}
	
	return ReLU.activate.(layer * weights)
end

# ╔═╡ e65644cb-b3ff-4e18-866e-61d27818edb7
"""
	forward_propagate(
		layer_in::Row,
		weight_mats::Vector{Matrix{R}}
	)::Vector{Matrix{R}} where {Row<:AbstractVector{<:Real}, R<:Real}

Perform forward propagation through the network.
"""
function forward_propagate(
	layer_in::Row,
	weight_mats::Vector{Matrix{R}}
)::Vector{Matrix{R}} where {Row<:AbstractVector{<:Real}, R<:Real}
	
	layer_out_prev::Matrix{R} =
		# layer_out = advance_layer(layer_in, weights)
		reduce(advance_layer, weight_mats[begin:end-1]; init = layer_in')
	
	layer_out::Matrix{R} = layer_out_prev * weight_mats[end]

	# subsequent_layers
	return [
		layer_out_prev,
		layer_out
	]
end

# ╔═╡ 717ae1b7-e5b6-497d-bcc5-d0c037097e07
let # Load supervised training data
	(inputs, goals) = get_inputs_and_goals()

	# Hyperparameter
	hidden_size = 4

	weight_mats::Vector{Matrix{Float64}} = init_rand_weight.([
		3 => hidden_size,
		hidden_size => 1
	])
	
	layer_in, goal = first.([inputs, goals])

	# Forward propagation
	layer_mid, layer_out = forward_propagate(layer_in, weight_mats)
	
	@info "Status:" layer_in layer_mid layer_out weight_mats
end

# ╔═╡ 4c8c2a7a-de99-4a03-a07f-685f1633cf5e
md"""
# Backpropagation in Code
"""

# ╔═╡ c3aab21b-589b-4b90-a89f-30d387ff7007
"""
	back_propagate(
		L_in::Row,
		subsequent_layers::Vector{Matrix{R}},
		ΔL_out::Matrix{R},
		weight_mats::Vector{Matrix{R}}
	)::Vector{Matrix{R}} where {Row<:AbstractVector{<:Real}, R<:Real}

Perform backpropagation to calculate weight updates.
"""
function back_propagate(
	L_in::Row,
	subsequent_layers::Vector{Matrix{R}},
	ΔL_out::Matrix{R},
	weight_mats::Vector{Matrix{R}}
)::Vector{Matrix{R}} where {Row<:AbstractVector{<:Real}, R<:Real}
	
	relu2_deriv::Function = ReLU.deriv_factory(.0)
	
	L1::Matrix{R} = subsequent_layers[end - 1]
	ΔL1::Matrix{R} = (ΔL_out * weight_mats[2]') .* relu2_deriv.(L1)
	
	# @debug "Propagated:" ΔL1
	
	# weight_Δs
	return [
		L_in * ΔL1,
		L1' * ΔL_out
	]
end

# ╔═╡ bca9f1e1-1de2-4a6f-9bd5-861fd9fafea5


# ╔═╡ 75a9be55-0dd0-47c0-84b2-32b87d797132
md"""
# One Iteration of Backpropagation
"""

# ╔═╡ 8a622db6-aff0-407d-a65b-aaca49e9b17d


# ╔═╡ f43339ba-0ad0-4d54-8b09-c2abceb69c4e
md"""
## Putting it all Together
"""

# ╔═╡ d2fb1ff7-5be0-42b5-9636-4468f4311e8f


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "a4de803ffbed71b36720c404a0d3a578794e2972"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"
"""

# ╔═╡ Cell order:
# ╟─c4ac87fe-fe8f-11f0-bd44-e7be41300a4b
# ╟─941e115e-307c-4aec-a6d6-5f90866ecc3e
# ╠═9961822e-8687-49fe-8e76-2d456334ee57
# ╟─34cabbbf-a02f-410d-9e0f-43493eec46f1
# ╠═226b0583-10a8-4fe7-a6c2-9fbd5136a4f1
# ╟─20f1e96c-613d-42f4-928a-3c4f033263e2
# ╠═0ecea74e-0443-4d1a-9843-b1a8bee23eb5
# ╟─e2ea141a-560e-4f68-be97-44dbfc749a0f
# ╟─309d4b06-274a-4390-83bf-c75cd722202e
# ╠═10369220-f87d-4f0b-9477-74ac7121b2f1
# ╟─9309ea1d-de47-4a98-9989-c3cfe50dc0a8
# ╠═3c2fa6d0-0e3b-4f7e-925b-1486e52b6388
# ╠═18836503-681e-41bf-b214-3db0af8d14a2
# ╟─43386f27-ada9-483d-bc18-9c3ddbff0503
# ╟─ce062beb-067f-4f5b-b81c-f310a8938630
# ╟─89f3c7c3-0c68-4c56-860f-eba8a5d7428d
# ╟─8514e517-debe-4381-a7c1-3dba9dc107ee
# ╟─e65644cb-b3ff-4e18-866e-61d27818edb7
# ╠═717ae1b7-e5b6-497d-bcc5-d0c037097e07
# ╟─4c8c2a7a-de99-4a03-a07f-685f1633cf5e
# ╟─c3aab21b-589b-4b90-a89f-30d387ff7007
# ╠═bca9f1e1-1de2-4a6f-9bd5-861fd9fafea5
# ╟─75a9be55-0dd0-47c0-84b2-32b87d797132
# ╠═8a622db6-aff0-407d-a65b-aaca49e9b17d
# ╟─f43339ba-0ad0-4d54-8b09-c2abceb69c4e
# ╠═d2fb1ff7-5be0-42b5-9636-4468f4311e8f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
