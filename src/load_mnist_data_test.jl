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

    @testset "transform_data function" begin
        # Create a small test dataset: 2 images of size 2x2 and corresponding labels
        mock_images = rand(UInt8, 2, 2, 2)  # 2 random 2x2 images
        mock_labels = 0:1                   # Corresponding labels

        flattened_images, one_hot_labels = transform_data(mock_images, mock_labels)

        # Expected flattened shape: (4, 2)
        @test size(flattened_images) == (4, 2)

        # Expected one-hot encoded shape: (2, 2)
        @test size(one_hot_labels) == (2, 2)

        # Check one-hot encoding correctness
        @test one_hot_labels[1, 1] == true   # Label 0
        @test one_hot_labels[2, 2] == true   # Label 1
        @test sum(one_hot_labels[:, 1]) == 1 # Only one true in first column
        @test sum(one_hot_labels[:, 2]) == 1 # Only one true in second column
    end

    @testset "load_data function" begin
        # Test loading training data
        train_images, train_labels = load_data(Val(:train))
        @test size(train_images, 1) == 784  # 28*28 pixels
        @test size(train_labels, 2) == 10   # 10 classes

        # Test loading test data
        test_image_set, test_label_set = load_data(Val(:test))
        @test size(test_image_set, 1) == 784   # 28*28 pixels
        @test size(test_label_set, 2) == 10    # 10 classes
    end

end # @testset "MNIST Data Loading and Preprocessing"
