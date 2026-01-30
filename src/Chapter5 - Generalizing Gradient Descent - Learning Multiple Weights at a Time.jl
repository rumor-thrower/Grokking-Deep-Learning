### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 0d41a6b0-fdb0-11f0-906e-6f448a888760
md"""
# Gradient Descent Learning with Multiple Inputs
"""

# ╔═╡ df5a2868-4486-469a-bcdd-d9fca23b9585
import LinearAlgebra

# ╔═╡ bb4d9e65-8cc5-4d31-b80a-5df1b8bf9663
w_sum::Function = LinearAlgebra.dot

# ╔═╡ 732caae4-75b7-4c8f-abb6-058527fedb71
function neural_network(input::Vector{R}, weight::Vector{R})::R where R<:Real
	return w_sum(input, weight) # predict
end

# ╔═╡ 9a6c6ed6-b96d-4e32-97ed-e2ad960f581a
function extract_input_and_label()::Tuple{Vector{Float64}, Int}
	toes =  [8.5, 9.5, 9.9, 9.0]
	wlrec = [0.65, 0.8, 0.8, 0.9]
	nfans = [1.2, 1.3, 0.5, 1.0]
	
	win_or_lose_binary = [1, 1, 0, 1]
	label = win_or_lose_binary[begin]
	
	# Input corresponds to every entry
	# for the first game of the season.
	input = [toes[begin], wlrec[begin], nfans[begin]]
	
	return (input, label)
end

# ╔═╡ 0d81a5bc-17f2-4830-8587-532ca9631d09
ele_mul(number::N, vector::Vector{N}) where N<:Real = number * vector

# ╔═╡ fc07f78d-34dd-4ba8-8478-3eeaa96ef3b2
function fit(input::Vector{R}, weights::Vector{R}, delta::R, alpha::R) where R<:Real
	weight_deltas = ele_mul(delta, input)
	weights -= alpha * weight_deltas
	return (weights, weight_deltas)
end

# ╔═╡ 3a33b320-3a13-4b8f-aaa2-d3d11c61d7ee
function gradient_descent(weights::Vector{R}, alpha::R) where R<:Real
	(input, label) = extract_input_and_label()
	
	pred = neural_network(input, weights)
	
	delta = neural_network(input, weights) - label
	@assert delta ≈ -.14
	error = delta ^ 2
	
	(weights, weight_deltas) = fit(input, weights, delta, alpha)
	
	@info "Status:" pred error delta weights weight_deltas
end

# ╔═╡ 7b07298d-801c-48fb-a0b9-060ec9b48ade
gradient_descent([.1, .2, -.1], .01)

# ╔═╡ 8742b3e1-d64d-40e3-b2fc-cb1f31fb66aa
md"""
# Let's Watch Several Steps of Learning
"""

# ╔═╡ 3c56a7b9-fabd-4e4e-817b-d83ff31a40f1


# ╔═╡ a746572b-14cf-42f8-bc05-6b34f44d8a6b
md"""
# Freezing One Weight - What Does It Do?
"""

# ╔═╡ ca06a804-15db-4ce6-927e-1b158ff7c9e0


# ╔═╡ 93ec1387-4a5b-4c88-81ec-e374b22f15cd
md"""
# Gradient Descent Learning with Multiple Outputs
"""

# ╔═╡ ea7fe6ff-81f4-4bd4-bf1a-120fe9053dcb


# ╔═╡ 04f8b253-3847-4578-b9cf-7bdc9c67f015
md"""
# Gradient Descent with Multiple Inputs & Outputs
"""

# ╔═╡ b4d2afab-531d-4241-b9bd-a779318c96f3


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "f352ceee806168c8ae38887a01d7bae6ca62470b"

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

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"
"""

# ╔═╡ Cell order:
# ╟─0d41a6b0-fdb0-11f0-906e-6f448a888760
# ╠═df5a2868-4486-469a-bcdd-d9fca23b9585
# ╠═bb4d9e65-8cc5-4d31-b80a-5df1b8bf9663
# ╠═732caae4-75b7-4c8f-abb6-058527fedb71
# ╠═9a6c6ed6-b96d-4e32-97ed-e2ad960f581a
# ╠═0d81a5bc-17f2-4830-8587-532ca9631d09
# ╠═fc07f78d-34dd-4ba8-8478-3eeaa96ef3b2
# ╠═3a33b320-3a13-4b8f-aaa2-d3d11c61d7ee
# ╠═7b07298d-801c-48fb-a0b9-060ec9b48ade
# ╟─8742b3e1-d64d-40e3-b2fc-cb1f31fb66aa
# ╠═3c56a7b9-fabd-4e4e-817b-d83ff31a40f1
# ╟─a746572b-14cf-42f8-bc05-6b34f44d8a6b
# ╠═ca06a804-15db-4ce6-927e-1b158ff7c9e0
# ╟─93ec1387-4a5b-4c88-81ec-e374b22f15cd
# ╠═ea7fe6ff-81f4-4bd4-bf1a-120fe9053dcb
# ╟─04f8b253-3847-4578-b9cf-7bdc9c67f015
# ╠═b4d2afab-531d-4241-b9bd-a779318c96f3
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
