# src/forward_propagte_test.jl
# Test file for forward_propagate function

using Test

include("forward_propagate.jl")

@testset "forward_propagate" begin
    layer_in = [0.5, 0.2, 0.1]
    weight_mats = [
        [0.4 0.6; 0.3 0.9; 0.8 0.2],
        [0.5; 0.7]
    ]
    expected_intermediate_layers = [
        [0.34, 0.5]
    ]
    expected_layer_out = [0.52]

    intermediate_layers, layer_out = forward_propagate(layer_in, weight_mats)

    @test length(intermediate_layers) == 1
    @test all(i -> all(isapprox.(intermediate_layers[i], expected_intermediate_layers[i])), eachindex(expected_intermediate_layers))
    @test isapprox(layer_out, expected_layer_out)
end
