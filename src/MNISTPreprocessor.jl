# src/MNISTPreprocessor.jl
# Preprocess the MNIST dataset for use in neural networks

using MLDatasets: MNIST

# Set up DataDeps to avoid download prompts
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

# Load MNIST dataset
trainset, testset =
	(MNIST(split = type, dir = "../dataset/$type") for type in [:train, :test])

# Get all data
X_train, y_train = trainset[:]
X_test, y_test = testset[:]

# Use a subset for faster prototyping
const images = X_train[:, :, begin:1000]
const labels::Vector{Int64} = y_train[begin:1000]

const test_images = X_test[:, :, begin:1000]
const test_labels::Vector{Int64} = y_test[begin:1000]

@info "Training set:" summary(X_train) summary(y_train) summary(images) summary(labels)
@info "Test set:" summary(X_test) summary(y_test) summary(test_images) summary(test_labels)
