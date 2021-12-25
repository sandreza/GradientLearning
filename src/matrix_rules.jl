using ChainRulesCore
using Random
import ChainRulesCore: frule, rrule
using LinearAlgebra
const RealOrComplex = Union{Real,Complex}


# Addition
function frule(
    (_, ΔA, ΔB),
    ::typeof(+),
    A::Array{<:RealOrComplex},
    B::Array{<:RealOrComplex},
)
    Ω = A + B
    ∂Ω = ΔA + ΔB
    return (Ω, ∂Ω)
end


# Multiplication
function frule(
    (_, ΔA, ΔB),
    ::typeof(*),
    A::Matrix{<:RealOrComplex},
    B::Matrix{<:RealOrComplex},
)
    Ω = A * B
    ∂Ω = ΔA * B + A * ΔB
    return (Ω, ∂Ω)
end

function rrule(::typeof(*), A::Matrix{<:RealOrComplex}, B::Matrix{<:RealOrComplex})
    function times_pullback(ΔΩ)
        ∂A = @thunk(ΔΩ * B')
        ∂B = @thunk(A' * ΔΩ)
        return (NoTangent(), ∂A, ∂B)
    end
    return A * B, times_pullback
end

# inverse 
function frule((_, ΔA), ::typeof(inv), A::Matrix{<:RealOrComplex})
    Ω = inv(A)
    ∂Ω = -Ω * ΔA * Ω
    return (Ω, ∂Ω)
end

function rrule(::typeof(inv), A::Matrix{<:RealOrComplex})
    Ω = inv(A)
    function inv_pullback(ΔΩ)
        ∂A = -Ω' * ΔΩ * Ω'
        return (NoTangent(), ∂A)
    end
    return Ω, inv_pullback
end

##
# Testing 
Random.seed!(1234)
N = 3
Ȧ = zeros(N, N) + I
Ḃ = zeros(N, N) + I
A = randn(N, N)
B = randn(N, N) 
b, ḃ = frule((nofields, Ȧ, Ḃ), *, A, B); 
b - A * B
ḃ - (Ȧ * B + Ḃ * A)