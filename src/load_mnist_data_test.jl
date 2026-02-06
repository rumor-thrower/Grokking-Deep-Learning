# src/load_mnist_data_test.jl
# Test suite for the MNIST data loading and preprocessing functions.

using Test

include("load_mnist_data.jl")

@testset "MNIST Data Loading and Preprocessing" begin

    @testset "flatten_images function" begin
        # Create a small test dataset: 2 images of size 2x2
        mock_images = rand(UInt8, 2, 2, 2)  # 2 random 2x2 images

        # Expected flattened shape: (4, 2)
        flattened = flatten_images(mock_images)
        @test size(flattened) == (4, 2)

        # Check that the contents are correctly flattened
        @test flattened[:, 1] == vec(mock_images[:, :, 1])
        @test flattened[:, 2] == vec(mock_images[:, :, 2])
    end

end # @testset "MNIST Data Loading and Preprocessing"
