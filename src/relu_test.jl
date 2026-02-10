# src/relu_test.jl
# Tests for the ReLU activation function in Julia

using Test

include("relu.jl")

@testset "ReLU Activation Function Tests" begin
    @testset "relu function tests" begin
         @test relu(-2.0) == 0.0
         @test relu(3.5) == 3.5
         @test relu(0) == 0
    end
    
    @testset "relu2deriv function tests" begin
        @testset "relu2deriv without fallback" begin
            relu2deriv::Function = relu2deriv_factory()

            @test relu2deriv(-0.5) == 0
            @test relu2deriv(0.5) == 1

            @test_throws ArgumentError relu2deriv(0.0)
        end

        @testset "relu2deriv with fallback" begin
            relu2deriv_with_fallback::Function = relu2deriv_factory(0.5)

            @test relu2deriv_with_fallback(-0.5) == 0
            @test relu2deriv_with_fallback(0.5) == 1
            @test relu2deriv_with_fallback(0.0) == 0.5
        end
    end
end
