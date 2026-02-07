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

    @testset "init_rand_weight" begin
        dims = 2 => 3
        weight_mat = init_rand_weight(dims)
        @test size(weight_mat) == (2, 3)
        @test all(-1 .<= w .<= 1 for w in weight_mat)
    end

end # @testset "Weight Initialization Tests"
