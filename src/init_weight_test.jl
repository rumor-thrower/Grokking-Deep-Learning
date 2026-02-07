# src/init_weight_test.jl
# Test suite for the weight initialization functions.

using Test

include("init_weight.jl")

@testset "Weight Initialization Tests" begin

    @testset "normalize_neg1_to_1" begin
        M = [0.0 0.5; 1.0 0.25]
        normalized_M = normalize_neg1_to_1(M)
        @test normalized_M == [-1.0 0.0; 1.0 -0.5]
    end

end # @testset "Weight Initialization Tests"
