using ChainRulesCore
using ChainRulesTestUtils

function two2three(x1::Float64, x2::Float64)
    return 1.0, 2.0 * x1, 3.0 * x2
end


function ChainRulesCore.frule((Δf, Δx1, Δx2), ::typeof(two2three), x1, x2)
    y = two2three(x1, x2)
    ∂y = Tangent{Tuple{Float64,Float64,Float64}}(ZeroTangent(), 2.0 * Δx1, 3.0 * Δx2)
    return y, ∂y
end

function ChainRulesCore.rrule(::typeof(two2three), x1, x2)
    y = two2three(x1, x2)
    function two2three_pullback(Ȳ)
        return (NoTangent(), 2.0 * Ȳ[2], 3.0 * Ȳ[3])
    end
    return y, two2three_pullback
end