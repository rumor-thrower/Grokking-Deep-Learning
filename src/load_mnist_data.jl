# src/load_mnist_data.jl
# Load and preprocess the MNIST dataset for machine learning tasks.

include("MNISTPreprocessor.jl")

@isdefined images
@isdefined labels
@isdefined test_images
@isdefined test_labels

using SparseArrays: sparse, SparseMatrixCSC  # for one-hot encoding

"""
    flatten_images(images::Array{R, 3})::Matrix{R} where R <: Real

Flatten a 3D array of images into a 2D matrix where each column is a flattened image.

# Examples
```jldoctest
julia> include(@___FILE__)
julia> images = rand(UInt8, 28, 28, 5)  # 5 random 28x28 images
julia> flattened = flatten_images(images)
julia> size(flattened)
(784, 5)
```
"""
function flatten_images(images::Array{R, 3})::Matrix{R} where R <: Real
	pixel_count::Int = size(images, 1) * size(images, 2)
	return reshape(images, pixel_count, :)
end

"""
    transform_data(
        images::Array{R, 3},
        labels::Vector{Int64}
    )::Tuple{SparseMatrixCSC{R, Int64}, SparseMatrixCSC{Bool, Int64}} where R <: Real

Transform the given images and labels into a flattened image matrix and one-hot encoded label matrix.

# Examples
```jldoctest
julia> include(@___FILE__)
julia> images = rand(UInt8, 28, 28, 5)  # 5 random 28x28 images
julia> labels = 0:9                     # Corresponding labels
julia> flattened_images, one_hot_labels = transform_data(images, labels)
julia> size(flattened_images)
(784, 5)
julia> size(one_hot_labels)
(10, 5)
```
"""
function transform_data(
    images::Array{R, 3},
    labels::L
)::Tuple{SparseMatrixCSC{R, Int64}, SparseMatrixCSC{Bool, Int64}} where {R <: Real, L <: AbstractVector{Int64}}
	flattened_images::SparseMatrixCSC{R, Int64} = sparse(flatten_images(images))
	one_hot_labels = sparse(1:size(flattened_images, 2), labels .+ 1, true)
	return flattened_images, one_hot_labels
end

# Load training and test data
load_data(::Val{:train}) = transform_data(images, labels)
load_data(::Val{:test}) = transform_data(test_images, test_labels)
