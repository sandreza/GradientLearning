using ChainRulesCore
using ChainRulesTestUtils
using BenchmarkTools

function two2three(x1::Float64, x2::Float64)
    return 1.0, 2.0 * x1, 3.0 * x2
end

function ChainRulesCore.frule((Δf, Δx1, Δx2), ::typeof(two2three), x1, x2)
    y = two2three(x1, x2)
    ∂y = Tangent{Tuple{Float64, Float64, Float64}}(
        ZeroTangent(),
        2.0 * Δx1,
        3.0 * Δx2,
    )
    return y, ∂y
end

function ChainRulesCore.rrule(::typeof(two2three), x1, x2)
    y = two2three(x1, x2)
    function two2three_pullback(Ȳ)
        return (NoTangent(), 2.0 * Ȳ[2], 3.0 * Ȳ[3])
    end
    return y, two2three_pullback
end

test_frule(two2three, 3.33, -7.77)

##
function two2three(x::AbstractArray)
    return 1.0, 2.0 * x[1], 3.0 * x[2]
end

function ChainRulesCore.frule((Δf, Δx), ::typeof(two2three), x)
    y = two2three(x)
    ∂y = Tangent{Tuple{Float64, Float64, Float64}}(
        ZeroTangent(),
        2.0 * Δx[1],
        3.0 * Δx[2],
    )
    return y, ∂y
end

function ChainRulesCore.rrule(::typeof(two2three), x)
    y = two2three(x)
    function two2three_pullback(Ȳ)
        return (NoTangent(), [2.0 * Ȳ[2], 3.0 * Ȳ[3]])
    end
    return y, two2three_pullback
end

vec = [3.33, -7.77]
two2three(vec)
test_frule(two2three, vec)
test_rrule(two2three, vec)

##
function ntwo2three(x::AbstractArray)
    return 1.0, 2.0 * x[1]^2, 3.0 * x[2]^3
end

function ChainRulesCore.frule((Δf, Δx), ::typeof(ntwo2three), x)
    y = ntwo2three(x)
    ∂y = Tangent{Tuple{Float64, Float64, Float64}}(
        ZeroTangent(),
        4.0 * x[1] * Δx[1],
        9.0 * x[2]^2 * Δx[2],
    )
    return y, ∂y
end

function ChainRulesCore.rrule(::typeof(ntwo2three), x)
    y = ntwo2three(x)
    function ntwo2three_pullback(Ȳ)
        return (NoTangent(), [4.0 * x[1] * Ȳ[2], 9.0 * x[2]^2 * Ȳ[3]])
    end
    return y, ntwo2three_pullback
end

vec = [3.33, -7.77]
two2three(vec)
test_frule(ntwo2three, vec)
test_rrule(ntwo2three, vec)

##
function lorenz(s; σ = 10, ρ = 28, β = 8 / 3)
    x, y, z = s

    ẋ = σ * (x - y)
    ẏ = -y + (ρ - z) * x
    ż = -β * z + x * y

    return [ẋ, ẏ, ż]
end

using Zygote
tmp = Zygote.jacobian(lorenz, [1.0, 2.0, 3.0])
@show tmp

function lorenz!(ṡ, s; σ = 10, ρ = 28, β = 8 / 3)
    x, y, z = s
    ṡ[1] = -σ * (x - y)
    ṡ[2] = -y + (ρ - z) * x
    ṡ[3] = -β * z + x * y
    return nothing
end

function jlorenz!(ṡ, s, sᵃ; σ = 10, ρ = 28, β = 8 / 3)
    x, y, z = s
    xᵃ, yᵃ, zᵃ = sᵃ
    ṡ[1] = σ * (xᵃ - yᵃ)
    ṡ[2] = -yᵃ + (ρ - zᵃ) * x - z * xᵃ
    ṡ[3] = -β * zᵃ + xᵃ * y + x * yᵃ
    return nothing
end

function jlorenz(s, sᵃ; σ = 10, ρ = 28, β = 8 / 3)
    return Zygote.jacobian(lorenz, s)[1] * sᵃ
end

ṡ = [1.0, 2.0, 3.0]
s = copy(ṡ)
y = copy(ṡ)

@benchmark lorenz!(ṡ, s)
@benchmark lorenz(s)

@benchmark jlorenz!(ṡ, s, y)
@benchmark jlorenz(s, y)

function step!(s, ṡ, s̃, rhs!, Δt)
    rhs!(ṡ, s)
    @. s̃ = s + Δt * ṡ
    @. s = s + Δt * 0.5 * ṡ
    rhs!(ṡ, s̃)
    @. s = s + Δt * 0.5 * ṡ
    return nothing
end

function step(s, rhs, Δt)
    ṡ = rhs(s)
    s̃ = s + Δt * ṡ
    s2 = rhs(s̃)
    return s + Δt * 0.5 * (ṡ + s2)
end

function stepᵃ!(s, ṡ, s̃, sᵃ, i, rhs!, Δt)
    rhs!(ṡ, s[i], sᵃ)
    @. s̃ = s + Δt * ṡ
    @. s = s + Δt * 0.5 * ṡ
    rhs!(ṡ, s[i + 1], s̃)
    @. s = s + Δt * 0.5 * ṡ
    return nothing
end

s̃ = copy(s)
@benchmark step!(s, ṡ, s̃, lorenz!, 0.1)
@benchmark step(s, lorenz, 0.1)

s .= [1.0, 1.0, 1.0]
step(s, lorenz, 0.001)

function evolve(s, N)
    y = s
    for _ in 1:N
        y = step(y, lorenz, 0.001)
    end
    return y
end

function evolve!(s, ṡ, s̃, N)
    for _ in 1:N
        step!(s, ṡ, s̃, lorenz!, 0.001)
    end
    return nothing
end

@benchmark evolve(s, 10)
@benchmark evolve!(s, ṡ, s̃, 10)
