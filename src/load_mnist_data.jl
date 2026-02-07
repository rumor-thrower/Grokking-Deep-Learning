# src/load_mnist_data.jl
# Load and preprocess the MNIST dataset for machine learning tasks.

include("MNISTPreprocessor.jl")

@isdefined images
@isdefined labels
@isdefined test_images
@isdefined test_labels
