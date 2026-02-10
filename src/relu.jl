# src/relu.jl
# Implementation of the ReLU activation function in Julia

"""
	relu(x::R)::R where R<:Real

Apply `ReLU` activation function.

# Returns
Maximum of input `x` and zero for input type `R`.

# Example
```jldoctest
julia> include(@__FILE__)
julia> relu(-2.0)
0.0
julia> relu(3.5)
3.5
julia> relu(0)
0
```
"""
function relu(x::R)::R where R<:Real
	return max(zero(R), x)
end

"""
    relu2deriv_factory()::Function

# Returns
Derivative function of `ReLU`. At `x = 0`, throws `ArgumentError`.

# Explanation
The derivative of `ReLU` is:
- `0` for `x < 0`
- `1` for `x > 0`
- undefined at `x = 0` (since the left-hand limit and right-hand limit do not agree).

# Example
```jldoctest
julia> include(@__FILE__)
julia> relu2deriv::Function = relu2deriv_factory()
julia> relu2deriv(-.5)
# 0
julia> relu2deriv(.5)
# 1
julia> relu2deriv(.0)
ERROR: ArgumentError: The derivative at x = 0.0 is undefined
[...]
```
"""
relu2deriv_factory()::Function =
    function relu2deriv(x::R) where R<:Real
        handle_zero(x::R) = throw(ArgumentError("The derivative at x = $x is undefined"))
        handler::Function = x |> iszero ? handle_zero : R ∘ !signbit
        return handler(x)
    end

"""
    relu2deriv_factory(fallback::R)::Function where R<:Real

Get the derivative function of `ReLU` which uses `fallback` at `x = 0`.

# Example
```jldoctest
julia> include(@__FILE__)
julia> relu2deriv::Function = relu2deriv_factory(.0)
julia> relu2deriv(-.5)
# 0
julia> relu2deriv(.5)
# 1
julia> relu2deriv(.0)
# 0.0
```
"""
function relu2deriv_factory(fallback::R)::Function where R<:Real
    relu_deriv(x::R)::R = ifelse(x |> iszero, fallback, !signbit(x))
    return relu_deriv
end
