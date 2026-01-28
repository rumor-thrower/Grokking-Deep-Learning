### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ a09ed848-fc3f-11f0-9326-eb498920348a
md"""
# Compare: Does our network make good predictions?
"""

# ╔═╡ 0961c6f5-a925-4744-93e5-9f0665f68852
let
	knob_weight = 0.5
	input = 0.5
	goal_pred = 0.8
	
	pred = input * knob_weight
	error = (pred - goal_pred) ^ 2
	@assert error ≈ 0.3025
end

# ╔═╡ 11166a19-00b7-4dd1-a7fe-4113f2b2b80c
md"""
# What's the Simplest Form of Neural Learning?
"""

# ╔═╡ 515ceacb-b7b9-48ff-9f89-d47cb1682ad1
md"""
## Learning using the Hot and Cold Method
"""

# ╔═╡ 73f71bf4-2a55-446f-8480-82c524452d96
# 1) An Empty Network

neural_network(input::F, weight::F) where F<:AbstractFloat = input * weight # predict

# ╔═╡ 30a38ca9-8041-4ca8-bccc-cb96475d1502
# 2) PREDICT: Making A Prediction And Evaluating Error

function evaluate_error(weight::F, lr::F)::F where F<:AbstractFloat
	number_of_toes = [8.5]
	win_or_lose_binary = [1] #(won!!!)
	
	input = number_of_toes[begin]
	label = win_or_lose_binary[begin]
	
	pred = neural_network(input, weight + lr)
	error = (pred - label) ^ 2
end

# ╔═╡ 11550b01-6068-4f02-ab7f-8a7a14971ba3
let weight = 0.1
	error = evaluate_error(weight, 0.0)
	
	# 3) COMPARE: Making A Prediction With a *Higher* Weight And Evaluating Error
	@assert evaluate_error(weight, 0.01) < error
	
	# 4) COMPARE: Making A Prediction With a *Lower* Weight And Evaluating Error
	@assert evaluate_error(weight, -0.01) > error
end

# ╔═╡ 87a9eac4-089f-42d6-a184-c71950fa4e1b
md"""
# Hot and Cold Learning
"""

# ╔═╡ 4c1b81b9-45a1-463c-9e71-a6a45fc97a7a
function fit(
	input::F,
	goal_prediction::F,
	weight::F,
	step_amount::F,
	iteration::Int = 0
)::F where F<:AbstractFloat

	prediction = input * weight

	if prediction ≈ goal_prediction
		return weight
	else
	    error = (prediction - goal_prediction) ^ 2

	    @info "Status:" error prediction
	    
	    up_error = muladd(input, weight + step_amount, -goal_prediction) ^ 2
	    down_error = muladd(input, weight - step_amount, -goal_prediction) ^ 2

	    weight += step_amount *
			(down_error < up_error
			? -1
			: down_error > up_error
			? +1
			: 0)

		fit(input, goal_prediction, weight, step_amount, iteration + 1)
	end
end

# ╔═╡ 9dbe1c28-d4dc-4627-a54a-5e7e256a7c14
fit(0.5, 0.8, 0.5, 0.001)

# ╔═╡ 212f2d35-a435-41f3-bd99-ee8e57f43851
md"""
# Calculating Both Direction and Amount from Error
"""

# ╔═╡ c1126792-5041-439b-b278-38bd66994d03
let
	weight = 0.5
	goal_pred = 0.8
	input = 0.5
	
	for _ in 1:20
	    pred = input * weight
	    error = (pred - goal_pred) ^ 2
	    direction_and_amount = (pred - goal_pred) * input
	    weight = weight - direction_and_amount
	
	    @info "Status:" error pred
	end
end

# ╔═╡ 5872d389-8337-451b-8805-1813f5b3cad6
md"""
# One Iteration of Gradient Descent
"""

# ╔═╡ 11a75675-fd2e-4d6b-86d4-6c11b9766552


# ╔═╡ 8e8c38cc-a228-41ed-a55e-6623508234f7
md"""
# Learning is just Reducing Error
"""

# ╔═╡ 07f42bd6-f9d4-4d6e-8ac9-a9ed80ab3094


# ╔═╡ 6e717e49-d4bc-4627-b0fd-85f61254b9cc
md"""
# Let's Watch Several Steps of Learning
"""

# ╔═╡ 720e28c1-6e99-4b17-ace3-35764cbce995


# ╔═╡ 34fb89b1-cbff-484e-baf8-d733b9e49439
md"""
# Why does this work? What really is weight_delta?
"""

# ╔═╡ b15a7c33-4205-49b1-b1cc-d1fc3a766751


# ╔═╡ d05250de-4c48-475e-abe3-d9006ff4a2bf
md"""
# How to use a Derivative to Learn
"""

# ╔═╡ 4fdd13ae-470b-4009-bcef-1370b447ce12


# ╔═╡ d5cb4920-37e6-4ba9-86d8-60a61fe8e588
md"""
# Breaking Gradient Descent
"""

# ╔═╡ 2ea0fee9-41c6-43de-a3d6-34fd63148b34


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
# ╟─a09ed848-fc3f-11f0-9326-eb498920348a
# ╠═0961c6f5-a925-4744-93e5-9f0665f68852
# ╟─11166a19-00b7-4dd1-a7fe-4113f2b2b80c
# ╟─515ceacb-b7b9-48ff-9f89-d47cb1682ad1
# ╠═73f71bf4-2a55-446f-8480-82c524452d96
# ╠═30a38ca9-8041-4ca8-bccc-cb96475d1502
# ╠═11550b01-6068-4f02-ab7f-8a7a14971ba3
# ╟─87a9eac4-089f-42d6-a184-c71950fa4e1b
# ╠═4c1b81b9-45a1-463c-9e71-a6a45fc97a7a
# ╠═9dbe1c28-d4dc-4627-a54a-5e7e256a7c14
# ╟─212f2d35-a435-41f3-bd99-ee8e57f43851
# ╠═c1126792-5041-439b-b278-38bd66994d03
# ╟─5872d389-8337-451b-8805-1813f5b3cad6
# ╠═11a75675-fd2e-4d6b-86d4-6c11b9766552
# ╟─8e8c38cc-a228-41ed-a55e-6623508234f7
# ╠═07f42bd6-f9d4-4d6e-8ac9-a9ed80ab3094
# ╟─6e717e49-d4bc-4627-b0fd-85f61254b9cc
# ╠═720e28c1-6e99-4b17-ace3-35764cbce995
# ╟─34fb89b1-cbff-484e-baf8-d733b9e49439
# ╠═b15a7c33-4205-49b1-b1cc-d1fc3a766751
# ╟─d05250de-4c48-475e-abe3-d9006ff4a2bf
# ╠═4fdd13ae-470b-4009-bcef-1370b447ce12
# ╟─d5cb4920-37e6-4ba9-86d8-60a61fe8e588
# ╠═2ea0fee9-41c6-43de-a3d6-34fd63148b34
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
