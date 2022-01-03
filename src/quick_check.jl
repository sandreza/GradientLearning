M = [-1 1 0 0
    0 -1 1 0
    0 0 -1 1
    1 0 0 -1
]
using LinearAlgebra

Λ, V = eigen(M)
Λᵀ, Vᵀ = eigen(M')