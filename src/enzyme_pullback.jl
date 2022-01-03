using Enzyme, BenchmarkTools, Zygote, Random

function mymul!(R, A, B)
    @assert axes(A, 2) == axes(B, 1)
    @inbounds @simd for i in eachindex(R)
        R[i] = 0
    end
    @inbounds for j in axes(B, 2), i in axes(A, 1)
        @inbounds @simd for k in axes(A, 2)
            R[i, j] += A[i, k] * B[k, j]
        end
    end
    nothing
end


A = rand(50, 30)
B = rand(30, 70)

R = zeros(size(A, 1), size(B, 2))
∂z_∂R = rand(size(R)...)  # Some gradient/tangent passed to us

∂z_∂A = zero(A)
∂z_∂B = zero(B)

@benchmark Enzyme.autodiff(mymul!, Const, Duplicated(R, ∂z_∂R), Duplicated(A, ∂z_∂A), Duplicated(B, ∂z_∂B))

@benchmark tmp = Zygote.pullback(*, A, B)[2](∂z_∂R)

∂z_∂A .= 0.0
∂z_∂B .= 0.0
Enzyme.autodiff(mymul!, Const, Duplicated(R, ∂z_∂R), Duplicated(A, ∂z_∂A), Duplicated(B, ∂z_∂B))
tmp = Zygote.pullback(*, A, B)[2](∂z_∂R)

maximum(abs.(tmp[1] - ∂z_∂A)) ≤ eps(10 * maximum(abs.(tmp[1])))
maximum(abs.(tmp[2] - ∂z_∂B)) ≤ eps(10 * maximum(abs.(tmp[2])))

##  Try it on the Lorenz equations 
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

function jᵀlorenz!(ṡ, s, sᵃ; σ = 10, ρ = 28, β = 8 / 3)
    x, y, z = s
    xᵃ, yᵃ, zᵃ = sᵃ
    ṡ[1] = -σ * xᵃ + (ρ - z) * yᵃ + y * zᵃ
    ṡ[2] = σ * xᵃ + -1 * yᵃ + x * zᵃ
    ṡ[3] = 0 * xᵃ + -x * yᵃ + -β * zᵃ
    return nothing
end

ṡ = zeros(3)
s = copy(ṡ)
dṡ = copy(s)
ds = copy(s)

Random.seed!(1234)
dṡ .= randn(3)
ṡ .= 0.0
s .= 0.0
ds .= 0.0
Enzyme.autodiff(lorenz!, Const, Duplicated(ṡ, dṡ), Duplicated(s, ds))
enzyme_answ = copy(ds)

Random.seed!(1234)
dṡ .= randn(3)

jᵀlorenz!(ṡ, s, dṡ; σ = 10, ρ = 28, β = 8 / 3)
analytic_answ = copy(ṡ)

norm(enzyme_answ - analytic_answ)
@benchmark Enzyme.autodiff(lorenz!, Const, Duplicated(ṡ, dṡ), Duplicated(s, ds))
@benchmark jᵀlorenz!(ṡ, s, dṡ; σ = 10, ρ = 28, β = 8 / 3)
