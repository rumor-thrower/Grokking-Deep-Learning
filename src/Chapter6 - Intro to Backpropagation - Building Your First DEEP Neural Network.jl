### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ c4ac87fe-fe8f-11f0-bd44-e7be41300a4b
md"""
# Creating a Matrix or Two in Julia
"""

# ╔═╡ 941e115e-307c-4aec-a6d6-5f90866ecc3e
function get_input_and_goal()::Tuple{BitVector, Bool}
	streetlights = BitMatrix([ 1 0 1 ;
							   0 1 1 ;
							   0 0 1 ;
							   1 1 1 ;
							   0 1 1 ;
							   1 0 1 ])
	
	walk_vs_stop = BitVector([ 0, 1, 0, 1, 1, 0 ])

	input::BitVector = streetlights[begin, :]
	goal::Bool = walk_vs_stop[begin]

	return (input, goal)
end


# ╔═╡ 20f1e96c-613d-42f4-928a-3c4f033263e2
md"""
## Building Our Neural Network
"""

# ╔═╡ 0ecea74e-0443-4d1a-9843-b1a8bee23eb5


# ╔═╡ e2ea141a-560e-4f68-be97-44dbfc749a0f
md"""
# Learning the whole dataset!
"""

# ╔═╡ 309d4b06-274a-4390-83bf-c75cd722202e


# ╔═╡ 9309ea1d-de47-4a98-9989-c3cfe50dc0a8
md"""
# Our First "Deep" Neural Network
"""

# ╔═╡ 3c2fa6d0-0e3b-4f7e-925b-1486e52b6388


# ╔═╡ 4c8c2a7a-de99-4a03-a07f-685f1633cf5e
md"""
# Backpropagation in Code
"""

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
# ╟─c4ac87fe-fe8f-11f0-bd44-e7be41300a4b
# ╟─941e115e-307c-4aec-a6d6-5f90866ecc3e
# ╟─20f1e96c-613d-42f4-928a-3c4f033263e2
# ╠═0ecea74e-0443-4d1a-9843-b1a8bee23eb5
# ╟─e2ea141a-560e-4f68-be97-44dbfc749a0f
# ╠═309d4b06-274a-4390-83bf-c75cd722202e
# ╟─9309ea1d-de47-4a98-9989-c3cfe50dc0a8
# ╠═3c2fa6d0-0e3b-4f7e-925b-1486e52b6388
# ╟─4c8c2a7a-de99-4a03-a07f-685f1633cf5e
# ╠═bca9f1e1-1de2-4a6f-9bd5-861fd9fafea5
# ╟─75a9be55-0dd0-47c0-84b2-32b87d797132
# ╠═8a622db6-aff0-407d-a65b-aaca49e9b17d
# ╟─f43339ba-0ad0-4d54-8b09-c2abceb69c4e
# ╠═d2fb1ff7-5be0-42b5-9636-4468f4311e8f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
