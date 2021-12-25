using ChainRulesCore

function foo(x)
    a = sin(x)
    b = 0.2 + a
    c = asin(b)
    return c
end

# Define rules (alternatively get them for free via `using ChainRules`)
@scalar_rule(sin(x), cos(x))
@scalar_rule(+(x, y), (1.0, 1.0))
@scalar_rule(asin(x), inv(sqrt(1 - x^2)))

#### Find dfoo/dx via rrules
#### First the forward pass, gathering up the pullbacks
x = 3;
a, a_pullback = rrule(sin, x); # a_pullback(x) = (NoTangent(), cos(x) * x)
b, b_pullback = rrule(+, 0.2, a);
c, c_pullback = rrule(asin, b)

#### Then the backward pass calculating gradients
c̄ = 1;                    # ∂c/∂c
_, b̄ = c_pullback(c̄);     # ∂c/∂b = ∂c/∂b ⋅ ∂c/∂c
println(b̄)
println(inv(sqrt(1 - b^2)))
_, ā2, ā = b_pullback(b̄);  # ∂c/∂a = ∂c/∂b ⋅ ∂b/∂a
println((ā2, ā))
println(b̄ .* (1.0, 1.0))
_, x̄ = a_pullback(ā);     # ∂c/∂x = ∂c/∂a ⋅ ∂a/∂x
println(x̄)                # ∂c/∂x = ∂foo/∂x
# ∂c/∂x = ∂c/∂b ⋅ ∂b/∂a ⋅ ∂a/∂x
#### Find dfoo/dx via frules
x = 3;
ẋ = 1;              # ∂x/∂x
nofields = ZeroTangent();  # ∂self/∂self

a, ȧ = frule((nofields, ẋ), sin, x);                    # ∂a/∂x = ∂a/∂x ⋅ ∂x/∂x 
println(ȧ)
println(cos(x))
b, ḃ = frule((nofields, ZeroTangent(), ȧ), +, 0.2, a);  # ∂b/∂x = ∂b/∂a ⋅ ∂a/∂x

c, ċ = frule((nofields, ḃ), asin, b);                   # ∂c/∂x = ∂c/∂b ⋅ ∂b/∂x
ċ                                                       # ∂c/∂x = ∂foo/∂x
# ∂c/∂x = ∂c/∂b ⋅ ∂b/∂a ⋅ ∂a/∂x
#### Find dfoo/dx via FiniteDifferences.jl
using FiniteDifferences
central_fdm(5, 1)(foo, x)
# output
-1.0531613736418257

#### Find dfoo/dx via ForwardDiff.jl
using ForwardDiff
ForwardDiff.derivative(foo, x)
# output
-1.0531613736418153

#### Find dfoo/dx via Zygote.jl
using Zygote
Zygote.gradient(foo, x)