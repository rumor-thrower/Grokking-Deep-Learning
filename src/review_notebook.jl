### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 62e817a2-fd0b-11f0-bb2e-534070a4b937
function grad_desc_factory(input::R, goal_pred::R, alpha::R)::Function where R<:Real
	max_f = prevfloat(typemax(R))
	mid_f = prevfloat(max_f)
	min_f = prevfloat(mid_f)

	return function grad_descent(
		weight::R,
		mse_pairs::Pair{R, Pair{R, R}} = max_f => mid_f => min_f
	)::R where R<:Real
		
		pred = input * weight
		if pred ≈ goal_pred
			weight
		else
			older_mse, old_mse = mse_pairs.second
			pure_error = pred - goal_pred
			new_mse = pure_error ^ 2
			if older_mse < old_mse < new_mse
				@error "Rising MSE, wrong alpha:" alpha
				# Also covers rocking MSE case
				return weight
			elseif older_mse ≈ old_mse ≈ new_mse
				@error "Fixed MSE, wrong alpha:" alpha
				return weight
			end
			derivative = input * pure_error
			new_weight = weight - alpha * derivative
			
			@info "Status:" pred new_mse
			
			grad_descent(new_weight, older_mse => old_mse => new_mse)
	   end
	end
end

# ╔═╡ e86a9100-22de-4de1-b447-c2b806e19bc2
grad_desc_factory(2.0, 0.8, 0.1)(0.5) # good alpha

# ╔═╡ 73d957d5-88ef-4593-b6b5-5b2118cc65cd
grad_desc_factory(2.0, 0.8, 0.0)(0.5) # bad alpha - no update

# ╔═╡ 419aad87-1767-451f-82a9-e9a2d4de6485
grad_desc_factory(2.0, 0.8, 0.5)(0.5) # bad alpha - no update

# ╔═╡ 1b973a40-6d38-4066-bb34-5604fa9e64b5
grad_desc_factory(2.0, 0.8, 1.0)(0.5) # bad alpha - divergence

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
# ╠═62e817a2-fd0b-11f0-bb2e-534070a4b937
# ╠═e86a9100-22de-4de1-b447-c2b806e19bc2
# ╠═73d957d5-88ef-4593-b6b5-5b2118cc65cd
# ╠═419aad87-1767-451f-82a9-e9a2d4de6485
# ╠═1b973a40-6d38-4066-bb34-5604fa9e64b5
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
