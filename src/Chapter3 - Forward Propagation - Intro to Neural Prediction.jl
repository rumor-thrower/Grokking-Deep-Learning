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

neural_network(input, weight = 0.1) = input * weight # prediction

# ╔═╡ 6e21e0ee-69fc-4398-a6dc-a812b4f4a209
# How we use the network to predict something:

let
	number_of_toes = [8.5, 9.5, 10, 9]
	input = number_of_toes[begin]
	pred = neural_network(input)
	@assert pred ≈ 0.85
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
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
