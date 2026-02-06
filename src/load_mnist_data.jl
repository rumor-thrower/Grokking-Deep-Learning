# src/load_mnist_data.jl
# Load and preprocess the MNIST dataset for machine learning tasks.

include("MNISTPreprocessor.jl")

@isdefined images
@isdefined labels
@isdefined test_images
@isdefined test_labels

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
