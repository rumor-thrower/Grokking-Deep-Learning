### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 5e080ed0-fbe5-11f0-9d1d-5b563764da62
md"""
# A Simple Neural Network Making a Prediction
"""

# ╔═╡ d0848d84-8c0b-4a83-9cb9-56b834a42559
md"""
## What is a Neural Network?
"""

# ╔═╡ 7ffdc80b-8c54-46b9-a485-643822b3d435
# The network:

neural_network(input::N, weight::F = 0.1) where {N<:Real, F<:AbstractFloat} =
	input * weight # prediction

# ╔═╡ 1649a784-2ec8-42fa-8a11-01259cb05b58
md"""
# Making a Prediction with Multiple Inputs
"""

# ╔═╡ 65889eb3-1e2b-45e0-9b8d-0ce244d20283
md"""
## Complete Runnable Code
"""

# ╔═╡ d53a4bff-8535-4481-98fe-c0b4489e5c61
function w_sum(a::Vector{N}, b::Vector{N}) where N<:Real
	@assert length(a) == length(b)
	output = zero(N)
	map(1:length(a)) do i
		output += a[i] * b[i]
	end
	return output
end

# ╔═╡ 8105e6f0-4aee-4d18-a644-1ed4c33503b7
neural_network(input::Vector{F}, weights::Vector{F} = [0.1, 0.2, 0]) where F<:AbstractFloat =
	w_sum(input, weights) # prediction

# ╔═╡ 6e21e0ee-69fc-4398-a6dc-a812b4f4a209
# How we use the network to predict something:

let
	number_of_toes = [8.5, 9.5, 10, 9]
	input = number_of_toes[begin]
	pred = neural_network(input)
	@assert pred ≈ 0.85
end

# ╔═╡ 2ddf7d1b-cb29-4b75-9681-4fa50d87eec7
let
	# This dataset is the current
	# status at the beginning of
	# each game for the first 4 games
	# in a season.
	
	# toes = current number of toes
	# wlrec = current games won (percent)
	# nfans = fan count (in millions)
	
	toes =  [8.5, 9.5, 9.9, 9.0]
	wlrec = [0.65, 0.8, 0.8, 0.9]
	nfans = [1.2, 1.3, 0.5, 1.0]
	
	# Input corresponds to every entry
	# for the first game of the season.
	input = [toes[begin], wlrec[begin], nfans[begin]]
	pred = neural_network(input)
	@assert pred ≈ 0.98
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "71853c6197a6a7f222db0f1978c7cb232b87c5ee"

[deps]
"""

# ╔═╡ Cell order:
# ╟─5e080ed0-fbe5-11f0-9d1d-5b563764da62
# ╟─d0848d84-8c0b-4a83-9cb9-56b834a42559
# ╠═7ffdc80b-8c54-46b9-a485-643822b3d435
# ╠═6e21e0ee-69fc-4398-a6dc-a812b4f4a209
# ╟─1649a784-2ec8-42fa-8a11-01259cb05b58
# ╟─65889eb3-1e2b-45e0-9b8d-0ce244d20283
# ╠═d53a4bff-8535-4481-98fe-c0b4489e5c61
# ╠═8105e6f0-4aee-4d18-a644-1ed4c33503b7
# ╠═2ddf7d1b-cb29-4b75-9681-4fa50d87eec7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
