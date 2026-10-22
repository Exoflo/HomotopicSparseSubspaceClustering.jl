# using LinearAlgebra: norm

"""
    ksparse(k ::Int) -> ::Bool
    ksparse(k ::Vector{Int}) -> ::Vector{Bool}

    Return an anonymous function that check if the sparsity of vector x is equal to k
    (A::Matrix,b::Vector,x::Vector) -> sum(x != 0) == k

    ------------------------------------------------------------------------
"""
function ksparse(k::Integer)
    return (A,b,x) -> sum(x .!= 0.0) == k
end

function ksparse(thrs::Union{AbstractVector{<:Integer},NTuple{D,<:Integer}}) where{D}
    return (A,b,x) -> sum(x .!= 0.0) .== thrs
end



"""
    relerror(e ::Real) -> ::Bool
    relerror(e ::Vector{Real}) -> ::Vector{Bool}

    Return an anonymous function that check if the relative error between ground truth b and reconstructed vector A*x is under threshold e (and hardcoded sparcity of x bellow 20)
    (A::Matrix,b::Vector,x::Vector) -> (||A*x - b|| / ||b||) < e or sum(x != 0) == 20

    ------------------------------------------------------------------------
"""
function relerror(e::Real) 
    return (A,b,x) -> (norm(A*x - b) <= e*norm(b) || sum(x .!= 0.0) == 20)
end
function relerror(thrs::Union{<:AbstractVector{<:Real},NTuple{D,<:Real}}) where{D}
    return (A,b,x) -> (norm(A*x - b)/norm(b) .<= thrs .|| sum(x .!= 0.0) == 20)
end



"""
    bwidth(e ::Real) -> ::Bool
    bwidth(e ::Vector{Real}) -> ::Vector{Bool}

    Return an anonymous function that check if the distance between ground truth b and reconstructed vector A*x is under threshold e*sqrt(D) (and hardcoded sparcity of x bellow 20)
    (A::Matrix,b::Vector,x::Vector) -> ||A*x - b|| < e*dim(b) or sum(x != 0) == 20

    ------------------------------------------------------------------------
"""
function bwidth(e::Real) 
    return (A,b,x) -> (norm(A*x - b) <= e*sqrt(length(b)) || sum(x .!= 0.0) == 20)
end
function bwidth(thrs::Union{<:AbstractVector{<:Real},NTuple{D,<:Real}}) where{D}
    return (A,b,x) -> (norm(A*x - b)/sqrt(length(b)) .<= thrs .|| sum(x .!= 0.0) == 20)
end

"""
    mxp(mp ::Vector{Real}) -> ::Bool
    mxp(mp ::Vector{Vector{Real}}) -> ::Vector{Bool}

    mp = [m,p]

    Return an anonymous function that check if the distance between ground truth b and reconstructed vector A*x is smaller than m*||b|| + p*sqrt(D) (and hardcoded sparcity of x bellow 20)
    (A::Matrix,b::Vector,x::Vector) -> ||A*x - b|| < (m*||b|| + p*dim(b)) or sum(x != 0) == 20

    ------------------------------------------------------------------------
"""
function mxp(mp::AbstractVector{<:Real}) 
    return (A,b,x) -> (norm(A*x - b) <= ([norm(b) sqrt(length(b))]*mp)[1] || sum(x .!= 0.0) == 20)
end
function mxp(thrs::AbstractVector{<:AbstractVector{<:Real}})
    return (A,b,x) -> (norm(A*x - b) .<= vcat([[norm(b) sqrt(length(b))]] .* thrs ...) .|| sum(x .!= 0.0) == 20)
end  #GrosBricolage




"""
    This function is a tool provided to test custom metric function
    On evaluation, metric should return a ::Bool

    function metric(A,b,x) -> Bool

    where : A is a DxN Matrix 
            b is a D-elements Vector
            x is a N-elements Vector
    ---------------------------------------------------
"""
function test_metric(metric::Function)

    # Common Case
    print("Common case : ")
    A = rand(10,10)
    b = rand(10)
    x = rand(10)

    out = metric(A,b,x)
    println(ifelse(out isa Bool,"ok","fail"))
    if !(out isa Bool); println("Warning : metric don't return any Boolean type !") end


    # Non-square matrix A
    print("Non-square matrix A : ")
    A = rand(30,10)
    b = rand(30)
    x = rand(10)

    out = metric(A,b,x)
    println(ifelse(out isa Bool,"ok","fail"))
    if !(out isa Bool); println("Warning : metric don't support non-square matrix !") end


    # Null-matrix A
    print("Null-matrix A : ")
    A = zeros(30,10)
    b = rand(30)
    x = rand(10)

    out = metric(A,b,x)
    println(ifelse(out isa Bool,"ok","fail"))
    if !(out isa Bool); println("Warning : metric don't support null-matrix as A entry !") end


    # Null-vector x
    print("Null-vector x : ")
    A = rand(30,10)
    b = rand(30)
    x = zeros(10)

    out = metric(A,b,x)
    println(ifelse(out isa Bool,"ok","fail"))
    if !(out isa Bool); println("Warning : metric don't support null-vector as x entry !") end


    # Null-vector b
    print("Null-vector b : ")
    A = rand(30,10)
    b = zeros(30)
    x = rand(10)

    out = metric(A,b,x)
    println(ifelse(out isa Bool,"ok","fail"))
    if !(out isa Bool); println("Warning : metric don't support null-vector as b entry !") end
end


"""
    This function is a tool provided to test custom broadcasted metric function
    On evaluation, metric should return either a ::BitVector or a ::Vector{Bool}

    function metric(A,b,x) -> Union{BitVector,Vector{Bool}}

    where : A is a DxN Matrix 
            b is a D-elements Vector
            x is a N-elements Vector
    ---------------------------------------------------
"""
function test_br_metric(metric::Function)

    # Common Case
    print("Common case : ")
    A = rand(10,10)
    b = rand(10)
    x = rand(10)

    out = metric(A,b,x)
    println(ifelse(out isa Union{BitVector,Vector{Bool}},"ok","fail"))
    if !(out isa Union{BitVector,Vector{Bool}}); println("Warning : metric don't return any Boolean type !") end


    # Non-square matrix A
    print("Non-square matrix A : ")
    A = rand(30,10)
    b = rand(30)
    x = rand(10)

    out = metric(A,b,x)
    println(ifelse(out isa Union{BitVector,Vector{Bool}},"ok","fail"))
    if !(out isa Union{BitVector,Vector{Bool}}); println("Warning : metric don't support non-square matrix !") end


    # Null-matrix A
    print("Null-matrix A : ")
    A = zeros(30,10)
    b = rand(30)
    x = rand(10)

    out = metric(A,b,x)
    println(ifelse(out isa Union{BitVector,Vector{Bool}},"ok","fail"))
    if !(out isa Union{BitVector,Vector{Bool}}); println("Warning : metric don't support null-matrix as A entry !") end


    # Null-vector x
    print("Null-vector x : ")
    A = rand(30,10)
    b = rand(30)
    x = zeros(10)

    out = metric(A,b,x)
    println(ifelse(out isa Union{BitVector,Vector{Bool}},"ok","fail"))
    if !(out isa Union{BitVector,Vector{Bool}}); println("Warning : metric don't support null-vector as x entry !") end


    # Null-vector b
    print("Null-vector b : ")
    A = rand(30,10)
    b = zeros(30)
    x = rand(10)

    out = metric(A,b,x)
    println(ifelse(out isa Union{BitVector,Vector{Bool}},"ok","fail"))
    if !(out isa Union{BitVector,Vector{Bool}}); println("Warning : metric don't support null-vector as b entry !") end
end

