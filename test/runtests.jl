# test/runtests.jl
# Test suite of the utility library `Grok` for the book "Grokking Deep Learning"
# Adapted from: JuliaWeb/HTTP.jl, JuliaData/DataFrames.jl
# Original sources:
# - https://github.com/JuliaWeb/HTTP.jl/blob/master/test/runtests.jl
# - https://github.com/JuliaData/DataFrames.jl/blob/main/test/runtests.jl

using Test

@testset "Book 'Grokking Deep Learning' utility library `Grok`" begin
    testfiles = [
        "load_mnist_data_test.jl",
        "init_weight_test.jl",
        "relu_test.jl",
        "propagate/fore/forward_propagate_test.jl",
        # Add test files above
    ]

    # Allow passing specific test files as command-line arguments
    testfiles = ifelse(isempty(ARGS), testfiles, ARGS)

    if Threads.nthreads() < 2
        @warn("Running tests with only 1 thread: correctness of parallel operations is not checked")
        for filename in testfiles
            println("Running $filename tests...")
            include(joinpath(@__DIR__, "..", "src", filename))
        end
    else
        @info("Running tests with $(Threads.nthreads()) threads")
        Threads.@threads for filename in testfiles
            println("Running $filename tests...")
            # Use joinpath for thread safety
            include(joinpath(@__DIR__, "..", "src", filename))
        end
    end

end # @testset "Book 'Grokking Deep Learning' utility library `Grok`"
